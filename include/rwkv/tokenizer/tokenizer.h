#include <string>
#include <memory>
#include <optional>
#include <unordered_map>
#include <regex>

struct PairHash{
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept;
};

class GPT2Tokenizer {

    using BPE = std::pair<std::string, std::string>;
    using BPERanks = std::unordered_map<BPE, size_t, PairHash>;
    using Encoder = std::unordered_map<std::string, int64_t>;
    using Decoder = std::unordered_map<int64_t, std::string>;

    static std::regex pattern;

public:

    static std::optional<GPT2Tokenizer> load(std::string_view vocab_file, std::string_view merges_file);

    std::vector<int64_t> encode(const std::string&);
    std::string decode(const std::vector<int64_t>&);
    std::vector<std::string> tokenize(const std::string&);

    size_t vocab_size() const noexcept { return m_encoder.size(); }
    
protected:

    GPT2Tokenizer() = default;

    BPERanks m_bpe_ranks;
    Encoder m_encoder;
    Decoder m_decoder;
    std::unordered_map<char, std::string> m_byte_encoder;
    std::unordered_map<std::string, char> m_byte_decoder;

private:
    std::vector<std::string> bpe(const std::string& token);
};