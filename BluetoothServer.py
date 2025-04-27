

def init_bluetooth_server():
    import bluetooth
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]
    uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"

    bluetooth.advertise_service(
        server_sock,
        "SmartSightBluetoothService",
        service_id=uuid,
        service_classes=[uuid, bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE]
    )

    print("Waiting for Bluetooth connection on port", port)
    client_sock, client_info = server_sock.accept()
    print("Accepted Bluetooth connection from", client_info)
    client_sock.send("Bluetooth connected. Hello!".encode())
    
    return client_sock, server_sock
