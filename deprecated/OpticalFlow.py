import numpy as np
import cv2
clahe = cv2.createCLAHE(clipLimit=25.0, tileGridSize=(6,6))
frameCount = 0

# Modes:
# optical_flow_lk - optical flow using Lucas method
# frame_subtract - simple subtraction
# obstacle_no_direction - trying to detect obstacles from dense optical flow.


def plot_image_dilation(img_thresh):
    ## apply a series of dilations
    #img_dilated = cv2.erode(img_thresh.copy(), np.ones((10, 10), np.uint8), iterations=2)
    img_dilated = cv2.dilate(img_thresh.copy(), np.ones((10, 10), np.uint8), iterations=4)
    return img_dilated

def plot_image_threshold(gray_image, method, threshold=150):

    # apply gaussian blur
    gray_image = cv2.GaussianBlur(gray_image, (7, 7), 0)

    if method == cv2.threshold:
        if threshold != 0:
            # apply binary thresholding
            T, img_thresh = method(gray_image, threshold, 255, cv2.THRESH_BINARY) # THRESH_BINARY_INV
            print(T)
        else:
            # apply binary thresholding
            T, img_thresh = method(gray_image, threshold, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU) # THRESH_OTSU # THRESH_BINARY # THRESH_BINARY_INV
            print(T)
    else:
        # apply adaptiveThreshold thresholding
        img_thresh = method(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, threshold, 10) # ADAPTIVE_THRESH_MEAN_C # ADAPTIVE_THRESH_GAUSSIAN_C

    # Apply mask to remove background
    # img_thresh = cv2.bitwise_and(gray_image, gray_image, mask=img_thresh) # original image with no background
    return img_thresh

def plot_image_connected(img_dilated, image):
    # Perform connected component analysis
    output = cv2.connectedComponentsWithStats(img_dilated, 4, cv2.CV_32S)
    (num_labels, labels, stats, centroids) = output

    # Create a copy of the original image to draw rectangles on
    img_with_rectangles = np.copy(img_dilated)
    img_copy = np.copy(image)

    # Iterate over each component (excluding background label 0)
    for label in range(1, num_labels):
        # Get the statistics for the current component
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        print("Area: ", area)

        # Area to keep
        # keepArea = 100000 < area < 300000

        # If True
        # if keepArea:
        print("[INFO] keeping connected component '{}'".format(label))
        # Draw a rectangle around the current component
        cv2.rectangle(img_with_rectangles, (left, top), (left + width, top + height), (255, 255, 255), 5)
        cv2.rectangle(img_copy, (left, top), (left + width, top + height), (255, 255, 255), 5)

    return img_copy


def optical_flow_lk(prev_frame, curr_frame):
    global frameCount
    global clahe
    frameCount += 1
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # diff = cv2.absdiff(prev_gray, curr_gray)

    # # Apply threshold to highlight motion areas
    # _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # return thresh


    # enhanced_prev_gray = prev_gray
    # enhanced_curr_gray = curr_gray


    enhanced_prev_gray = clahe.apply(prev_gray)
    enhanced_curr_gray = clahe.apply(curr_gray)
    
    # Detect good features to track
    feature_params = dict(maxCorners=500, qualityLevel=0.15, minDistance=12, blockSize=6)
    prev_pts = cv2.goodFeaturesToTrack(enhanced_prev_gray, mask=None, **feature_params)
    if prev_pts is not None and len(prev_pts) > 0:
        # if frameCount % 25 == 0:
        #     print(f"Number of points detected: {len(prev_pts)}")
        # Define Lucas-Kanade parameters
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Calculate optical flow
        curr_pts, st, err = cv2.calcOpticalFlowPyrLK(enhanced_prev_gray, enhanced_curr_gray, prev_pts, None, **lk_params)

        # Select valid points
        good_prev_pts = prev_pts[st == 1]
        good_curr_pts = curr_pts[st == 1]
    
        # Draw arrows showing movement direction
        for i, (prev, curr) in enumerate(zip(good_prev_pts, good_curr_pts)):
            x0, y0 = prev.ravel()
            x1, y1 = curr.ravel()
            cv2.arrowedLine(enhanced_curr_gray, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 6)
    else:
        print("No features to track in previous frame.")
    return enhanced_curr_gray

def frame_subtract(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)

    # Apply threshold to highlight motion areas
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    return thresh

import cv2
import numpy as np

def calculate_foe_with_visual(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, 
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Get flow components
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Create a grid of coordinates
    h, w = flow_x.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Solve least squares to estimate FOE
    A = np.stack([flow_x.ravel(), flow_y.ravel()], axis=1)
    b = np.sum(A * np.stack([x.ravel(), y.ravel()], axis=1), axis=1)
    foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Draw the FOE point on the frame
    output_frame = next_frame.copy()
    foe_point = tuple(np.round(foe).astype(int))
    cv2.circle(output_frame, foe_point, radius=8, color=(0, 0, 255), thickness=-1)

    return output_frame

def calculate_foe(prev_frame, next_frame):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Get flow components
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Create a grid of coordinates
    h, w = flow_x.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Solve least squares to find the intersection point (FoE)
    A = np.stack([flow_x.ravel(), flow_y.ravel()], axis=1)
    b = np.sum(A * np.stack([x.ravel(), y.ravel()], axis=1), axis=1)
    foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    return tuple(foe)

def flow_to_foe(flow):
    # Get flow components
    flow_x, flow_y = flow[..., 0], flow[..., 1]

    # Create a grid of coordinates
    h, w = flow_x.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Solve least squares to find the intersection point (FoE)
    A = np.stack([flow_x.ravel(), flow_y.ravel()], axis=1)
    b = np.sum(A * np.stack([x.ravel(), y.ravel()], axis=1), axis=1)
    foe, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return tuple(foe)

def optical_flow_dense(prev_frame, curr_frame):
    """
    Computes dense optical flow between two frames using the Farneback algorithm.

    Args:
        prev_frame (numpy.ndarray): The previous frame (grayscale or color).
        curr_frame (numpy.ndarray): The current frame (grayscale or color).

    Returns:
        numpy.ndarray: A 2-channel array with optical flow vectors (dx, dy) for each pixel.
    """


    # Convert frames to grayscale if they are not already
    # Optical flow algorithms typically operate on single-channel (grayscale) images.
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    prev_gray = clahe.apply(prev_gray)
    curr_gray = clahe.apply(curr_gray)

    # Calculate dense optical flow using the Farneback method.
    # The Farneback algorithm computes flow for all pixels in the image.
    # Parameters explained:
    #   prev_gray: First 8-bit single-channel input image.
    #   curr_gray: Second 8-bit single-channel input image of the same size and type as prev_gray.
    #   None: Optional output flow image; if passed, it must have a size equal to prev_gray image size
    #         and type CV_32FC2.
    #   0.5: pyr_scale - parameter, specifying the image scale (less than 1) to build pyramids for each image;
    #                   a typical value is 0.5, which means each next pyramid layer is twice as small as the previous.
    #   3: levels - number of pyramid layers including the initial image; 1 means no pyramids are used
    #               (single-layer optical flow).
    #   15: winsize - averaging window size; larger values give more robust, but less detailed flow.
    #   3: iterations - number of iterations the algorithm does at each pyramid level.
    #   5: poly_n - size of the pixel neighborhood used to find polynomial expansion in each pixel;
    #               larger values mean the pixels are approximated by a smoother polynomial (more robust flow).
    #   1.2: poly_sigma - standard deviation of the Gaussian that is used to smooth derivatives used to find
    #                     polynomial expansion; can be 1.1 for poly_n = 5 and 1.5 for poly_n = 7.
    #   0: flags - operation flags;
    #              OPTFLOW_USE_INITIAL_FLOW (if not 0) means the input flow is used as an initial approximation.
    #              OPTFLOW_FARNEBACK_GAUSSIAN (if 0) means uses a Gaussian filter instead of a box filter.
    
    # Create mask
    mask = np.zeros_like(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV))
    # Set image saturation to maximum value as we do not need it
    mask[..., 1] = 255
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return mask


def optical_flow_dense(prev_frame, curr_frame):
    """
    Computes dense optical flow between two frames using the Farneback algorithm.

    Args:
        prev_frame (numpy.ndarray): The previous frame (grayscale or color).
        curr_frame (numpy.ndarray): The current frame (grayscale or color).

    Returns:
        numpy.ndarray: A 2-channel array with optical flow vectors (dx, dy) for each pixel.
    """


    # Convert frames to grayscale if they are not already
    # Optical flow algorithms typically operate on single-channel (grayscale) images.
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    prev_gray = clahe.apply(prev_gray)
    curr_gray = clahe.apply(curr_gray)

    # Calculate dense optical flow using the Farneback method.
    # The Farneback algorithm computes flow for all pixels in the image.
    # Parameters explained:
    #   prev_gray: First 8-bit single-channel input image.
    #   curr_gray: Second 8-bit single-channel input image of the same size and type as prev_gray.
    #   None: Optional output flow image; if passed, it must have a size equal to prev_gray image size
    #         and type CV_32FC2.
    #   0.5: pyr_scale - parameter, specifying the image scale (less than 1) to build pyramids for each image;
    #                   a typical value is 0.5, which means each next pyramid layer is twice as small as the previous.
    #   3: levels - number of pyramid layers including the initial image; 1 means no pyramids are used
    #               (single-layer optical flow).
    #   15: winsize - averaging window size; larger values give more robust, but less detailed flow.
    #   3: iterations - number of iterations the algorithm does at each pyramid level.
    #   5: poly_n - size of the pixel neighborhood used to find polynomial expansion in each pixel;
    #               larger values mean the pixels are approximated by a smoother polynomial (more robust flow).
    #   1.2: poly_sigma - standard deviation of the Gaussian that is used to smooth derivatives used to find
    #                     polynomial expansion; can be 1.1 for poly_n = 5 and 1.5 for poly_n = 7.
    #   0: flags - operation flags;
    #              OPTFLOW_USE_INITIAL_FLOW (if not 0) means the input flow is used as an initial approximation.
    #              OPTFLOW_FARNEBACK_GAUSSIAN (if 0) means uses a Gaussian filter instead of a box filter.
    
    # Create mask
    mask = np.zeros_like(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV))
    # Set image saturation to maximum value as we do not need it
    mask[..., 1] = 255
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Set image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Set image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return mask

def cvtGrey(prev_frame, curr_frame):
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    prev_gray = clahe.apply(prev_gray)
    curr_gray = clahe.apply(curr_gray)
    return prev_gray, curr_gray

def getFlow(prev_frame, curr_frame):
    height, width, _ = curr_frame.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    prev_grey, curr_gray = cvtGrey(prev_frame, curr_frame)
    # Farneback function
    flow = cv2.calcOpticalFlowFarneback(prev_grey, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def denseOF_HSV(prev_frame, curr_frame):
    height, width, _ = curr_frame.shape
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    prev_grey, curr_gray = cvtGrey(prev_frame, curr_frame)
    # Farneback function
    flow = cv2.calcOpticalFlowFarneback(prev_grey, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Get magnitude and angle of vector
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #x #y

    # Hue
    hsv[..., 0] = (angle * 180 / (np.pi))
    
    # Value: Intensity
    motion_thresh = 5  # adjust this based on your scene's scale and sensitivity
    max_value = 255
    sensitivity = 1
    ret, thresholded_mag = cv2.threshold(magnitude, motion_thresh, max_value, cv2.THRESH_TOZERO)
    mag_val = thresholded_mag * sensitivity
    # norm_mag = cv2.normalize(thresholded_mag, None, 0, 255, cv2.NORM_MINMAX)
    # norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = mag_val
    # hsv[..., 2] = magnitude * 255 * Sensitivity

    return hsv

def obstacle_no_direction(prev_frame, curr_frame):
    hsv = denseOF_HSV(prev_frame, curr_frame)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshed = plot_image_threshold(grey, cv2.threshold, 35)
    dialated = plot_image_dilation(threshed)
    plotted = plot_image_connected(dialated, curr_frame)
    return plotted # cv2.cvtColor(threshed, cv2.COLOR_HSV2BGR) # cv2.imdecode(plotted, cv2.IMREAD_COLOR)

def obstacle_and_TTC(prev_frame, curr_frame):
    flow = getFlow(prev_frame, curr_frame)
    (stuff1, stuff2) = flow_to_foe(flow)
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # threshed = plot_image_threshold(grey, cv2.threshold, 35)
    # dialated = plot_image_dilation(threshed)
    # plotted = plot_image_connected(dialated, curr_frame)
    return stuff1 # cv2.cvtColor(threshed, cv2.COLOR_HSV2BGR) # cv2.imdecode(plotted, cv2.IMREAD_COLOR)
