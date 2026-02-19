import pickle
import datetime
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QByteArray, QDataStream, QIODevice
from PyQt5.QtNetwork import QTcpServer, QTcpSocket, QHostAddress
import queue
import logging

class ServerWorker(QObject):
    """Worker that handles server operations in background thread"""
    connection_status = pyqtSignal(str)
    data_received = pyqtSignal(list)
    start_signal = pyqtSignal(object)
    stop_signal = pyqtSignal()
    finished = pyqtSignal()
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.server = None
        self.client_socket = None
        self.host_ip = self.parent.defaults["tcp_connection"]["host_addr"]
        self.port = self.parent.defaults["tcp_connection"].getint("port")
        self.server_active = False
        
        # Command queue for main thread to send commands to worker
        self.cmd_queue = queue.Queue()
        
        # Timer intervals
        self.cmd_interval = 0.1  # Check commands every 100ms
        self.update_status_interval = 2  # Update camera status interval

        self.buffer = QByteArray()
    
    @pyqtSlot()
    def process_commands(self):
        """Process commands from main thread"""

        if not self.cmd_queue.empty():
            try:
                cmd, vals = self.cmd_queue.get_nowait()
                
            except queue.Empty:
                pass


    def process_received_objects(self, received_objects):
        # For each element in the list of already received objects, we process accordingly
        # The received message should follow the same pattern as in the generate_message method 
        for data_object in received_objects:
            if not isinstance(data_object, dict) or 'type' not in data_object:
                logging.error("Data object not legal. It is: ", data_object)
                return  # or raise Exception, etc.         
            
            message_type = data_object["type"]
            if message_type == "command":
                if data_object["payload"] == "Stop":
                    self.stop_signal.emit()
            
            elif message_type == "sequence info":
                seq_info = data_object["payload"]
                with open(self.parent.defaults["scan_file_name"]["default"], "w") as f:
                    seq_info.write(f)

                timestamp = seq_info["general"].get("timestamp", "")
                local_folder = self.parent.defaults.get("scan_file_name", "local_folder")
                if timestamp and local_folder:
                    local_filename = f"{local_folder}\\_{timestamp}.ini"
                    with open(local_filename, "w") as f:
                        seq_info.write(f)
                else:
                    logging.warning("Timestamp or local folder not found; skipping local save.")

                # Then, we turn on the camera
                self.start_signal.emit(seq_info)
                confirm_message = self.generate_message("confirm", "Received")
                success = self.send_data(confirm_message)

                if not success:
                    logging.error(f"Server failed to confirm the sequence instruction.")


    # Update the latest status of the camera
    def update_server_status(self):
        if not self.client_socket or not self.server_active:
            return

        status = "Running" if self.parent.control.active else "Idle"
        status_message = self.generate_message("status", status)
        self.send_data(status_message)

    def generate_message(self, type, payload):
        message = {
            "type": type,
            "timestamp": datetime.datetime.now(),
            "payload": payload
        }
        return message


    # Usually, you shouldn't need to modify, or even read the generic server methods 
    # They handle generic functionality for a QTcpServer,including start, stop, hook a new client, send data and receive data

    ########################################################
    # ------------ GENERIC SERVER METHODS -----------------#
    ########################################################

    def start_server(self):
        """Start the server on specified port"""
        
        try:
            self.server = QTcpServer()
            if self.server.listen(QHostAddress(self.host_ip), self.port):
                self.server_active = True
                self.connection_status.emit(f"Listening on port {self.port}")
                
                # Connect server signals
                self.server.newConnection.connect(self.handle_new_connection)
            else:
                error_msg = f"Failed to start server: {self.server.errorString()}"
                self.connection_status.emit(f"Error: {error_msg}")
                logging.error(error_msg)
                self.server = None
                self.server_active = False
                
        except Exception as e:
            error_msg = f"Exception starting server: {str(e)}"
            self.connection_status.emit(f"Error: {error_msg}")
            logging.eror(error_msg)
            self.server_active = False
    
    @pyqtSlot()
    def stop_server(self):
        """Stop the server and clean up"""
        self.server_active = False
        
        # Handle client socket
        if self.client_socket:
            try:
                self.client_socket.readyRead.disconnect()
                self.client_socket.disconnected.disconnect()
                self.client_socket.error.disconnect()
            except Exception as e:
                print("Error in disconnecting client socket. Error is ", e)
                pass
            
            if self.client_socket.state() == QTcpSocket.ConnectedState:
                self.client_socket.disconnectFromHost()
                if self.client_socket.state() != QTcpSocket.UnconnectedState:
                    self.client_socket.waitForDisconnected(1000)
            
            self.client_socket.deleteLater()
            self.client_socket = None

        # Handle server
        if self.server:
            try:
                self.server.newConnection.disconnect()
            except Exception as e:
                print("Error in disconnecting server. Error is ", e)
                pass
            
            self.server.close()
            self.server.deleteLater()
            self.server = None
        
        self.connection_status.emit("Server stopped")
        self.finished.emit()

    
    def handle_new_connection(self):
        """Handle new client connections"""
        if not self.server_active:
            return
            
        self.client_socket = self.server.nextPendingConnection()
        if self.client_socket:
            client_info = f"{self.client_socket.peerAddress().toString()}:{self.client_socket.peerPort()}"
            self.connection_status.emit(f"Connected client {client_info}")
            
            # Connect socket signals
            self.client_socket.readyRead.connect(self.receive_data)
            self.client_socket.disconnected.connect(self.handle_client_disconnected)
            self.client_socket.error.connect(self.handle_socket_error)
    
    def handle_client_disconnected(self):
        """Handle client disconnection"""
        self.connection_status.emit("Client disconnected")
        self.client_socket = None
        
    def handle_socket_error(self, error):
        """Handle socket errors"""
        if self.client_socket:
            error_msg = self.client_socket.errorString()
            self.connection_status.emit(f"Socket error: {error_msg}")
            # logging.error(f"Socket error: {error_msg}")

    # Send any data (python object) to socket
    def send_data(self, pyobj):
        """Send data via socket"""
        if not self.client_socket or not self.server_active:
            logging.error("Cannot send data: Socket not connected.")
            return False
            
        if self.client_socket.state() != QTcpSocket.ConnectedState:
            logging.error("Cannot send data: Socket not connected.")
            return False
            
        try:
            data = pickle.dumps(pyobj)
            block = QByteArray()
            stream = QDataStream(block, QIODevice.WriteOnly)
            stream.setVersion(QDataStream.Qt_5_0)
            stream.writeUInt32(len(data))
            stream.writeRawData(data)
            bytes_written = self.client_socket.write(block)
            self.client_socket.flush()
            
            if bytes_written == -1:
                logging.error(f"Failed to send data {pyobj}")
                return False
            else:
                return True     # Indicating success in sending data
                
        except Exception as e:
            logging.error(f"Error sending data: {e}")
            return False
    
    # Read from socket buffer, and pack all received python objects into a list and emit the list to a processing method
    def receive_data(self):
        """Receive data, append it to a persistent buffer, and process all complete objects"""
        if not self.client_socket or not self.server_active:
            return

        data = self.client_socket.readAll()
        if data.size() == 0:
            return
        
        self.buffer.append(data)
        received_objects = []
        stream = QDataStream(self.buffer, QIODevice.ReadOnly)
        stream.setVersion(QDataStream.Qt_5_0)
        
        while not stream.atEnd():
            start_pos = stream.device().pos()

            if self.buffer.size() - start_pos < 4:
                break

            size = stream.readUInt32()

            if self.buffer.size() - stream.device().pos() < size:
                stream.device().seek(start_pos)
                break
                
            try:    
                serialized_data = stream.readRawData(size)
                received_obj = pickle.loads(serialized_data)
                received_objects.append(received_obj)
            except Exception as e:
                logging.error(f"Deserialization error: {e}. Clearing buffer.")
                self.buffer.clear()
                break

        final_pos = int(stream.device().pos())
        if final_pos > 0:
            self.buffer = self.buffer.right(self.buffer.size() - final_pos)

        if received_objects:
            self.data_received.emit(received_objects)