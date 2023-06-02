#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstring>
#include <algorithm>

// include to_integer
#include <iostream>
#include <string>
#include <iterator>
#include <numeric>
#include <cstddef>
std::string from_bstring( std::string strValue) {
   
    // Create byte array of same size as string length
    std::byte byteArr[strValue.length()];
    // Iterate over characters in string,
    // convert them to byte and copy to byte array
    std::transform(
        strValue.begin(),
        strValue.end(),
        byteArr,
        [](const char& ch) {
            return std::byte(ch);
        });
    // Iterate over byte array and print
    // for (const std::byte& byt: byteArr) 
    // {
    //     std::cout << std::to_integer<int>(byt) << ", ";
    // }
    // std::cout<<std::endl;
    // return 0;
    
    // Create string from byte array
    std::string out = std::string(reinterpret_cast<char*>(byteArr), sizeof(byteArr));
}

class RWKV_TOKENIZER {
private:
    std::unordered_map<int, std::string> idx2token;
    std::unordered_map<std::string, int> token2idx;
    std::vector<int> wlen;

public:
    RWKV_TOKENIZER(const std::string& file_name) {
        std::vector<std::vector<char>> sorted; // must be already sorted
        std::ifstream file(file_name);
        std::string line;

        while (std::getline(file, line)) {
            int idx = std::stoi(line.substr(0, line.find(' ')));
            std::string x_str = line.substr(line.find(' ')+1, line.length());
            // cut to last thing thats not a number or a space
            x_str = x_str.substr(0, x_str.find_last_not_of("0123456789 "));
            //x_str = x_str.substr(0, x_str.find(' '));

            // Check if the string is in the format b'...'
            if (x_str[0] == 'b') {
                idx2token[idx] = from_bstring(x_str.substr(2, x_str.length()));
            }else{
                idx2token[idx] = x_str.substr(1, x_str.length());
                // interprate /n as newline and all other escaped characters
                for (int i = 0; i < idx2token[idx].length(); i++) {
                    if (idx2token[idx][i] == '\\') {
                        switch (idx2token[idx][i + 1]) {
                        case 'n':
                            idx2token[idx][i] = '\n';
                            break;
                        case 't':
                            idx2token[idx][i] = '\t';
                            break;
                        case 'r':
                            idx2token[idx][i] = '\r';
                            break;
                        case 'b':
                            idx2token[idx][i] = '\b';
                            break;
                        case 'f':
                            idx2token[idx][i] = '\f';
                            break;
                        case '\\':
                            idx2token[idx][i] = '\\';
                            break;
                        case '\'':
                            idx2token[idx][i] = '\'';
                            break;
                        case '\"':
                            idx2token[idx][i] = '\"';
                            break;
                        case '0':
                            idx2token[idx][i] = '\0';
                            break;
                        }
                        idx2token[idx].erase(i + 1, 1);
                    }
                }
            }
            token2idx[idx2token[idx]] = idx;
            
        }

        file.close();


    }

    std::string decodeBytes(const std::vector<long>& tokens) {
        std::string result;
        for (int i : tokens) {
            std::string s = idx2token[i];
            result += s;
        }
        return result;
    }

    std::vector<long long> encode(const std::string& src) {
        std::vector<long long> result;
        std::string s = src;
        while (s.length() > 0) {
            int startlen = 22;
            if (s.length() < startlen) {
                startlen = s.length();
            }
            while (startlen > 0) {
                std::string sub = s.substr(0, startlen);
                if (token2idx.find(sub) != token2idx.end()) {
                    result.push_back(token2idx[sub]);
                    s = s.substr(startlen, s.length());
                    break;
                }
                startlen--;
            }
        }
        return result;        
    }

    std::string decode(const std::vector<long>& tokens) {
        return decodeBytes(tokens);
    }

    void printTokens(const std::vector<int>& tokens) {
        for (int i : tokens) {
            std::string s = idx2token[i];
            std::cout << "\"" << s << "\"" << i << " ";
        }
        std::cout << std::endl;
    }
};

