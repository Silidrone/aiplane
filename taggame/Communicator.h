#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>
#include <string>

class Communicator {
   public:
    const std::string RESET = "reset";

    static Communicator& getInstance() {
        static Communicator instance;
        return instance;
    }

    bool connectToServer(const std::string& host, int port) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Socket creation failed." << std::endl;
            return false;
        }

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &serverAddr.sin_addr);

        if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "Connection to server failed." << std::endl;
            close(sock);
            return false;
        }

        std::cout << "Connected to server at " << host << ":" << port << std::endl;
        return true;
    }

    void disconnect() {
        if (sock >= 0) {
            close(sock);
            sock = -1;
            std::cout << "Disconnected from server." << std::endl;
        }
    }

    std::string receiveState() {
        char buffer[4096];
        ssize_t bytesReceived = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (bytesReceived > 0) {
            buffer[bytesReceived] = '\0';
            return std::string(buffer);
        } else if (bytesReceived == 0) {
            std::cerr << "Server closed the connection." << std::endl;
        } else {
            std::cerr << "Error receiving state." << std::endl;
        }
        return "";
    }

    void sendAction(const std::string& action) {
        std::string actionWithNewline = action + "\n";  // Add newline for Java `readLine`
        ssize_t bytesSent = send(sock, actionWithNewline.c_str(), actionWithNewline.size(), 0);
        if (bytesSent < 0) {
            std::cerr << "Error sending action." << std::endl;
        }
    }

   private:
    int sock = -1;

    Communicator() = default;
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    ~Communicator() { disconnect(); }
};