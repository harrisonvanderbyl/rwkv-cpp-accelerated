#include "simdjson.h"
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <fstream>
#include <optional>



struct TokPairHash {
	std::size_t operator()(const std::pair<std::string, std::string> &p) const noexcept ;
};


template <class T>
inline void hash_combine(std::size_t &seed, const T &v) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::unordered_map<char, std::string> bytes_to_unicode() {
	// Because I have no idea what I am doing the code_map was copy pasted from bytes_to_unicode()
	// definetely not crossplatform, but works on POSIX maybe
	static std::unordered_map<char, std::string> code_map = { { 33, "!" }, { 34, "\"" }, { 35, "#" }, { 36, "$" }, { 37, "%" }, { 38, "&" }, { 39, "\'" }, { 40, "(" }, { 41, ")" }, { 42, "*" }, { 43, "+" }, { 44, "," }, { 45, "-" }, { 46, "." }, { 47, "/" }, { 48, "0" }, { 49, "1" }, { 50, "2" }, { 51, "3" }, { 52, "4" }, { 53, "5" }, { 54, "6" }, { 55, "7" }, { 56, "8" }, { 57, "9" }, { 58, ":" }, { 59, ";" }, { 60, "<" }, { 61, "=" }, { 62, ">" }, { 63, "?" }, { 64, "@" }, { 65, "A" }, { 66, "B" }, { 67, "C" }, { 68, "D" }, { 69, "E" }, { 70, "F" }, { 71, "G" }, { 72, "H" }, { 73, "I" }, { 74, "J" }, { 75, "K" }, { 76, "L" }, { 77, "M" }, { 78, "N" }, { 79, "O" }, { 80, "P" }, { 81, "Q" }, { 82, "R" }, { 83, "S" }, { 84, "T" }, { 85, "U" }, { 86, "V" }, { 87, "W" }, { 88, "X" }, { 89, "Y" }, { 90, "Z" }, { 91, "[" }, { 92, "\\" }, { 93, "]" }, { 94, "^" }, { 95, "_" }, { 96, "`" }, { 97, "a" }, { 98, "b" }, { 99, "c" }, { 100, "d" }, { 101, "e" }, { 102, "f" }, { 103, "g" }, { 104, "h" }, { 105, "i" }, { 106, "j" }, { 107, "k" }, { 108, "l" }, { 109, "m" }, { 110, "n" }, { 111, "o" }, { 112, "p" }, { 113, "q" }, { 114, "r" }, { 115, "s" }, { 116, "t" }, { 117, "u" }, { 118, "v" }, { 119, "w" }, { 120, "x" }, { 121, "y" }, { 122, "z" }, { 123, "{" }, { 124, "|" }, { 125, "}" }, { 126, "~" }, { 161, "¡" }, { 162, "¢" }, { 163, "£" }, { 164, "¤" }, { 165, "¥" }, { 166, "¦" }, { 167, "§" }, { 168, "¨" }, { 169, "©" }, { 170, "ª" }, { 171, "«" }, { 172, "¬" }, { 174, "®" }, { 175, "¯" }, { 176, "°" }, { 177, "±" }, { 178, "²" }, { 179, "³" }, { 180, "´" }, { 181, "µ" }, { 182, "¶" }, { 183, "·" }, { 184, "¸" }, { 185, "¹" }, { 186, "º" }, { 187, "»" }, { 188, "¼" }, { 189, "½" }, { 190, "¾" }, { 191, "¿" }, { 192, "À" }, { 193, "Á" }, { 194, "Â" }, { 195, "Ã" }, { 196, "Ä" }, { 197, "Å" }, { 198, "Æ" }, { 199, "Ç" }, { 200, "È" }, { 201, "É" }, { 202, "Ê" }, { 203, "Ë" }, { 204, "Ì" }, { 205, "Í" }, { 206, "Î" }, { 207, "Ï" }, { 208, "Ð" }, { 209, "Ñ" }, { 210, "Ò" }, { 211, "Ó" }, { 212, "Ô" }, { 213, "Õ" }, { 214, "Ö" }, { 215, "×" }, { 216, "Ø" }, { 217, "Ù" }, { 218, "Ú" }, { 219, "Û" }, { 220, "Ü" }, { 221, "Ý" }, { 222, "Þ" }, { 223, "ß" }, { 224, "à" }, { 225, "á" }, { 226, "â" }, { 227, "ã" }, { 228, "ä" }, { 229, "å" }, { 230, "æ" }, { 231, "ç" }, { 232, "è" }, { 233, "é" }, { 234, "ê" }, { 235, "ë" }, { 236, "ì" }, { 237, "í" }, { 238, "î" }, { 239, "ï" }, { 240, "ð" }, { 241, "ñ" }, { 242, "ò" }, { 243, "ó" }, { 244, "ô" }, { 245, "õ" }, { 246, "ö" }, { 247, "÷" }, { 248, "ø" }, { 249, "ù" }, { 250, "ú" }, { 251, "û" }, { 252, "ü" }, { 253, "ý" }, { 254, "þ" }, { 255, "ÿ" }, { 0, "Ā" }, { 1, "ā" }, { 2, "Ă" }, { 3, "ă" }, { 4, "Ą" }, { 5, "ą" }, { 6, "Ć" }, { 7, "ć" }, { 8, "Ĉ" }, { 9, "ĉ" }, { 10, "Ċ" }, { 11, "ċ" }, { 12, "Č" }, { 13, "č" }, { 14, "Ď" }, { 15, "ď" }, { 16, "Đ" }, { 17, "đ" }, { 18, "Ē" }, { 19, "ē" }, { 20, "Ĕ" }, { 21, "ĕ" }, { 22, "Ė" }, { 23, "ė" }, { 24, "Ę" }, { 25, "ę" }, { 26, "Ě" }, { 27, "ě" }, { 28, "Ĝ" }, { 29, "ĝ" }, { 30, "Ğ" }, { 31, "ğ" }, { 32, "Ġ" }, { 127, "ġ" }, { 128, "Ģ" }, { 129, "ģ" }, { 130, "Ĥ" }, { 131, "ĥ" }, { 132, "Ħ" }, { 133, "ħ" }, { 134, "Ĩ" }, { 135, "ĩ" }, { 136, "Ī" }, { 137, "ī" }, { 138, "Ĭ" }, { 139, "ĭ" }, { 140, "Į" }, { 141, "į" }, { 142, "İ" }, { 143, "ı" }, { 144, "Ĳ" }, { 145, "ĳ" }, { 146, "Ĵ" }, { 147, "ĵ" }, { 148, "Ķ" }, { 149, "ķ" }, { 150, "ĸ" }, { 151, "Ĺ" }, { 152, "ĺ" }, { 153, "Ļ" }, { 154, "ļ" }, { 155, "Ľ" }, { 156, "ľ" }, { 157, "Ŀ" }, { 158, "ŀ" }, { 159, "Ł" }, { 160, "ł" }, { 173, "Ń" } };
	return code_map;
}

std::unordered_map<std::string, char> unicode_to_bytes() {
	static std::unordered_map<std::string, char> code_map = { { "!", 33 }, { "\"", 34 }, { "#", 35 }, { "$", 36 }, { "%", 37 }, { "&", 38 }, { "\'", 39 }, { "(", 40 }, { ")", 41 }, { "*", 42 }, { "+", 43 }, { ",", 44 }, { "-", 45 }, { ".", 46 }, { "/", 47 }, { "0", 48 }, { "1", 49 }, { "2", 50 }, { "3", 51 }, { "4", 52 }, { "5", 53 }, { "6", 54 }, { "7", 55 }, { "8", 56 }, { "9", 57 }, { ":", 58 }, { ";", 59 }, { "<", 60 }, { "=", 61 }, { ">", 62 }, { "?", 63 }, { "@", 64 }, { "A", 65 }, { "B", 66 }, { "C", 67 }, { "D", 68 }, { "E", 69 }, { "F", 70 }, { "G", 71 }, { "H", 72 }, { "I", 73 }, { "J", 74 }, { "K", 75 }, { "L", 76 }, { "M", 77 }, { "N", 78 }, { "O", 79 }, { "P", 80 }, { "Q", 81 }, { "R", 82 }, { "S", 83 }, { "T", 84 }, { "U", 85 }, { "V", 86 }, { "W", 87 }, { "X", 88 }, { "Y", 89 }, { "Z", 90 }, { "[", 91 }, { "\\", 92 }, { "]", 93 }, { "^", 94 }, { "_", 95 }, { "`", 96 }, { "a", 97 }, { "b", 98 }, { "c", 99 }, { "d", 100 }, { "e", 101 }, { "f", 102 }, { "g", 103 }, { "h", 104 }, { "i", 105 }, { "j", 106 }, { "k", 107 }, { "l", 108 }, { "m", 109 }, { "n", 110 }, { "o", 111 }, { "p", 112 }, { "q", 113 }, { "r", 114 }, { "s", 115 }, { "t", 116 }, { "u", 117 }, { "v", 118 }, { "w", 119 }, { "x", 120 }, { "y", 121 }, { "z", 122 }, { "{", 123 }, { "|", 124 }, { "}", 125 }, { "~", 126 }, { "¡", 161 }, { "¢", 162 }, { "£", 163 }, { "¤", 164 }, { "¥", 165 }, { "¦", 166 }, { "§", 167 }, { "¨", 168 }, { "©", 169 }, { "ª", 170 }, { "«", 171 }, { "¬", 172 }, { "®", 174 }, { "¯", 175 }, { "°", 176 }, { "±", 177 }, { "²", 178 }, { "³", 179 }, { "´", 180 }, { "µ", 181 }, { "¶", 182 }, { "·", 183 }, { "¸", 184 }, { "¹", 185 }, { "º", 186 }, { "»", 187 }, { "¼", 188 }, { "½", 189 }, { "¾", 190 }, { "¿", 191 }, { "À", 192 }, { "Á", 193 }, { "Â", 194 }, { "Ã", 195 }, { "Ä", 196 }, { "Å", 197 }, { "Æ", 198 }, { "Ç", 199 }, { "È", 200 }, { "É", 201 }, { "Ê", 202 }, { "Ë", 203 }, { "Ì", 204 }, { "Í", 205 }, { "Î", 206 }, { "Ï", 207 }, { "Ð", 208 }, { "Ñ", 209 }, { "Ò", 210 }, { "Ó", 211 }, { "Ô", 212 }, { "Õ", 213 }, { "Ö", 214 }, { "×", 215 }, { "Ø", 216 }, { "Ù", 217 }, { "Ú", 218 }, { "Û", 219 }, { "Ü", 220 }, { "Ý", 221 }, { "Þ", 222 }, { "ß", 223 }, { "à", 224 }, { "á", 225 }, { "â", 226 }, { "ã", 227 }, { "ä", 228 }, { "å", 229 }, { "æ", 230 }, { "ç", 231 }, { "è", 232 }, { "é", 233 }, { "ê", 234 }, { "ë", 235 }, { "ì", 236 }, { "í", 237 }, { "î", 238 }, { "ï", 239 }, { "ð", 240 }, { "ñ", 241 }, { "ò", 242 }, { "ó", 243 }, { "ô", 244 }, { "õ", 245 }, { "ö", 246 }, { "÷", 247 }, { "ø", 248 }, { "ù", 249 }, { "ú", 250 }, { "û", 251 }, { "ü", 252 }, { "ý", 253 }, { "þ", 254 }, { "ÿ", 255 }, { "Ā", 0 }, { "ā", 1 }, { "Ă", 2 }, { "ă", 3 }, { "Ą", 4 }, { "ą", 5 }, { "Ć", 6 }, { "ć", 7 }, { "Ĉ", 8 }, { "ĉ", 9 }, { "Ċ", 10 }, { "ċ", 11 }, { "Č", 12 }, { "č", 13 }, { "Ď", 14 }, { "ď", 15 }, { "Đ", 16 }, { "đ", 17 }, { "Ē", 18 }, { "ē", 19 }, { "Ĕ", 20 }, { "ĕ", 21 }, { "Ė", 22 }, { "ė", 23 }, { "Ę", 24 }, { "ę", 25 }, { "Ě", 26 }, { "ě", 27 }, { "Ĝ", 28 }, { "ĝ", 29 }, { "Ğ", 30 }, { "ğ", 31 }, { "Ġ", 32 }, { "ġ", 127 }, { "Ģ", 128 }, { "ģ", 129 }, { "Ĥ", 130 }, { "ĥ", 131 }, { "Ħ", 132 }, { "ħ", 133 }, { "Ĩ", 134 }, { "ĩ", 135 }, { "Ī", 136 }, { "ī", 137 }, { "Ĭ", 138 }, { "ĭ", 139 }, { "Į", 140 }, { "į", 141 }, { "İ", 142 }, { "ı", 143 }, { "Ĳ", 144 }, { "ĳ", 145 }, { "Ĵ", 146 }, { "ĵ", 147 }, { "Ķ", 148 }, { "ķ", 149 }, { "ĸ", 150 }, { "Ĺ", 151 }, { "ĺ", 152 }, { "Ļ", 153 }, { "ļ", 154 }, { "Ľ", 155 }, { "ľ", 156 }, { "Ŀ", 157 }, { "ŀ", 158 }, { "Ł", 159 }, { "ł", 160 }, { "Ń", 173 } };
	return code_map;
}

std::size_t TokPairHash::operator()(const std::pair<std::string, std::string> &p) const noexcept {
	std::size_t seed = 0;
	hash_combine(seed, p.first);
	hash_combine(seed, p.second);
	return seed;
}

class GPT2Tokenizer {
	using BPE = std::pair<std::string, std::string>;
	using BPERanks = std::unordered_map<BPE, size_t, TokPairHash>;
	using Encoder = std::unordered_map<std::string, int64_t>;
	using Decoder = std::unordered_map<int64_t, std::string>;

public:

	const std::regex pattern = std::regex("'s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s[:alpha:][:digit:]]+|\\s+(?!\\S)|\\s+");


	static std::optional<GPT2Tokenizer> load(std::string_view vocab_file, std::string_view merges_file) {
		// load merges file
		std::ifstream merges_file_stream;
		// assuming null-terminated string
		merges_file_stream.open(merges_file.data());

		if (!merges_file_stream.good()) {
			std::cerr << "Error: could not open merges file " << merges_file << std::endl;
			return std::nullopt;
		}

		BPERanks bpe_ranks;

		std::string merges_version;
		std::getline(merges_file_stream, merges_version);

		for (struct {std::string line; size_t i{0}; } it; std::getline(merges_file_stream, it.line); ++it.i) {
			const size_t split_point = it.line.find(' ');
			std::pair<std::string, std::string> p{ { it.line.begin(), it.line.begin() + split_point },
				{ it.line.begin() + split_point + 1, it.line.end() } };
			bpe_ranks.emplace(std::move(p), it.i);
		}

		simdjson::dom::parser parser;
		simdjson::dom::object object;
		// assuming null-terminated string
		simdjson::dom::element doc = parser.load(vocab_file.data());

		auto error = doc.get(object);
		if (error) {
			std::cerr << "Error: " << error << std::endl;
			return std::nullopt;
		}

		Encoder encoder;
		Decoder decoder;

		for (const auto &[key, value] : object) {
			encoder.emplace(key, value);
			decoder.emplace(value, key);
		}

		auto result = GPT2Tokenizer();
		result.m_bpe_ranks = std::move(bpe_ranks);
		result.m_encoder = std::move(encoder);
		result.m_decoder = std::move(decoder);
		result.m_byte_encoder = bytes_to_unicode();
		result.m_byte_decoder = unicode_to_bytes();

		return result;
	}

	std::vector<int64_t> encode(const std::string &text) {
		std::vector<std::string> tokens = tokenize(text);
		std::vector<int64_t> token_ids;
		token_ids.reserve(tokens.size());
		std::transform(tokens.begin(), tokens.end(), std::back_inserter(token_ids),
				[this](const std::string &token) {
					return m_encoder[token];
				});
		return token_ids;
	}

	size_t codepoint_length(const char c) {
		if ((c & 0xf8) == 0xf0)
			return 4;
		else if ((c & 0xf0) == 0xe0)
			return 3;
		else if ((c & 0xe0) == 0xc0)
			return 2;
		else
			return 1;
	}

	std::string decode(const std::vector<int64_t> &token_ids) {
		std::string decoded_string;
		for (const auto &id : token_ids) {
			std::string decoded_token = m_decoder[id];
			for (size_t i = 0; i < decoded_token.size();) {
				int length = codepoint_length(decoded_token[i]);
				decoded_string += m_byte_decoder[decoded_token.substr(i, length)];
				i += length;
			}
		}
		return decoded_string;
	}
	std::vector<std::string> tokenize(const std::string &text) {
		std::vector<std::string> result;
		auto words_begin =
				std::sregex_iterator(text.begin(), text.end(), pattern);
		auto words_end = std::sregex_iterator();

		for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
			std::smatch match = *i;
			std::string token = match.str();
			std::string byte_token;
			for (const auto &t : token) {
				byte_token += m_byte_encoder[t];
			}
			std::vector<std::string> bpe_result = bpe(byte_token);
			result.reserve(result.size() + bpe_result.size());
			result.insert(result.end(), bpe_result.begin(), bpe_result.end());
		}

		return result;
	}

	size_t vocab_size() const noexcept { return m_encoder.size(); }

protected:
	GPT2Tokenizer() = default;

	BPERanks m_bpe_ranks;
	Encoder m_encoder;
	Decoder m_decoder;
	std::unordered_map<char, std::string> m_byte_encoder;
	std::unordered_map<std::string, char> m_byte_decoder;

private:
	std::vector<std::string> bpe(const std::string &token) {
		std::vector<BPERanks::const_iterator> ranks;
		std::vector<std::string> word;
		ranks.reserve(token.size() - 1);
		word.reserve(token.size());

		// this essentially avoids having literal spaces ' ' in a string
		// at the same time we fetch the ranks of the bigrams
		{
			size_t i = 0;
			while (true) {
				int length = codepoint_length(token[i]);
				int next_length = codepoint_length(token[i + length]);
				ranks.push_back(
						m_bpe_ranks.find({ token.substr(i, length), token.substr(i + length, next_length) }));
				word.push_back(token.substr(i, length));
				i += length;
				if (i >= token.size())
					break;
				if (i + next_length >= token.size()) {
					word.emplace_back(token.substr(i, next_length));
					break;
				}
			}
		}

		while (true) {
			const auto bigram = std::min_element(ranks.begin(), ranks.end(),
					[this](const auto &lhs, const auto &rhs) -> bool {
						if (lhs == m_bpe_ranks.end() && lhs == m_bpe_ranks.end()) {
							return false;
						} else if (lhs == m_bpe_ranks.end() || rhs == m_bpe_ranks.end()) {
							return (lhs != m_bpe_ranks.end());
						} else {
							return lhs->second < rhs->second;
						}
					});
			if (*bigram == m_bpe_ranks.end()) {
				// could not find any matches in ranks
				break;
			}
			const auto [first, second] = (*bigram)->first;
			std::vector<std::string> new_word;

			size_t i = 0;
			while (i < word.size()) {
				const auto wordIterator = std::find(word.begin() + i, word.end(), first);
				if (wordIterator == word.end()) {
					std::copy(word.begin() + i, word.end(), std::back_inserter(new_word));
					break;
				}

				std::copy(word.begin() + i, wordIterator, std::back_inserter(new_word));
				i = std::distance(word.begin(), wordIterator);

				if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
					new_word.push_back(first + second);
					i += 2;
				} else {
					new_word.push_back(word[i]);
					i += 1;
				}
			}
			word = std::move(new_word);
			if (word.size() == 1)
				break;
			else {
				for (size_t i = 0; i < word.size() - 1; ++i) {
					ranks[i] = m_bpe_ranks.find({ word[i], word[i + 1] });
				}
				ranks.resize(word.size() - 1);
			}
		}

		return word;
	}
};