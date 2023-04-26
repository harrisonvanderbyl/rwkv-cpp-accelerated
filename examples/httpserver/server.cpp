#include <iostream>
#include <boost/asio.hpp>
#include "rwkv.h"
// TODO actually use this to create a rwkv server

using boost::asio::ip::tcp;

const std::string HTTP_OK_RESPONSE = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";

class HttpServer {
public:
    HttpServer(boost::asio::io_context& io_context, short port)
        : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
        do_accept();
    }

private:
    void do_accept() {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket) {
                if (!ec) {
                    std::make_shared<HttpSession>(std::move(socket))->start();
                }

                do_accept();
            });
    }

    class HttpSession : public std::enable_shared_from_this<HttpSession> {
    public:
        explicit HttpSession(tcp::socket socket)
            : socket_(std::move(socket)) {}

        void start() {
            do_read();
        }

    private:
        void do_read() {
            auto self(shared_from_this());
            socket_.async_read_some(
                buffer_,
                [this, self](boost::system::error_code ec, std::size_t length) {
                    if(ec){
                        std::cout << ec.message() << std::endl;
                    }
                    if (!ec) {
                        std::string data((char*)buffer_.data(), length);
                        std::cout << data << std::endl;

                        // Check if the request is a POST request
                        if (data.find("POST") != std::string::npos) {
                            std::cout << "POST request received" << std::endl;
                            // Handle the POST request
                            handle_post_request(data);
                        }
                        
                        do_read();
                    }
                });
        }

        void handle_post_request(const std::string& data) {
            // Here you can process the POST request data and send a response back to the client
            // In this example, we just send a HTTP OK response
            boost::asio::write(socket_, boost::asio::buffer(HTTP_OK_RESPONSE));
        }

        tcp::socket socket_;
        boost::asio::mutable_buffer buffer_ = boost::asio::buffer(new char[1024], 1024);
    };

    tcp::acceptor acceptor_;
};

int main() {
    try {
        boost::asio::io_context io_context;
        HttpServer server(io_context, 8080);
        io_context.run();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
