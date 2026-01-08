#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <unicode/utypes.h>
#include <unicode/unistr.h>
#include <unicode/translit.h>
#include <nlohmann/json.hpp>
#include "WordPiece.h"


using json = nlohmann::json;

namespace nlp::tokenizer {

    WordPiece::WordPiece(
        const std::string& config_path,  // Eg: "tokenizer.json".
        const std::string& vocab_key,  // Eg: "/model/vocab".
        bool lowercase,
        bool strip_accents,
        bool clean_text,
        bool handle_chinese_chars,
        std::size_t max_length
    ) : vocab_list_(std::make_unique<VocabList>()) {

        this->lowercase = lowercase;
        this->strip_accents = strip_accents;
        this->clean_text = clean_text;
        this->handle_chinese_chars = handle_chinese_chars;
        this->max_length = max_length;

        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open config file: " << config_path << std::endl;
            exit(-1);
        }

        json config;
        try {
            file >> config;
            auto vocab = config.at(json::json_pointer(vocab_key));

            for (auto& [token, id] : vocab.items()) {
                uint32_t token_id = id.get<uint32_t>();
                if (!vocab_list_->set_token(token, token_id)) {
                    std::cerr << "Warning: Could not set token '" << token << "' with ID " << token_id << std::endl;
                }
            }
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            exit(-1);
        }
    }

    TokenRole WordPiece::identify_special_token(uint32_t id) const {
        return TokenRole::None;
    }

    std::vector<Token> WordPiece::encode(std::string_view text) const {
        std::string normalised_text(text);  // Local copy to work with.
        if (lowercase) to_lowercase_inplace(normalised_text);
        if (strip_accents) strip_accents_inplace(normalised_text);
        if (clean_text) clean_text_inplace(normalised_text);
        if (handle_chinese_chars) handle_chinese_chars_inplace(normalised_text);

        std::cout << "normalised text:" << normalised_text << std::endl;

        // todo

        return {};
    }

    void WordPiece::build_byte_encoder() {

    }

    std::vector<std::string> WordPiece::bpe_merge(std::string_view word) const {

        return {};
    }

    std::vector<std::string_view> WordPiece::pre_tokenize(std::string_view text) const {

        return {};
    }

    void WordPiece::to_lowercase_inplace(std::string& text) const {
        std::ranges::transform(text, text.begin(),
            [](unsigned char c) { return std::tolower(c); });
    }

    void WordPiece::clean_text_inplace(std::string& text) const {

    }

    void WordPiece::strip_accents_inplace(std::string& text) const {
        UErrorCode status = U_ZERO_ERROR;
        icu::UnicodeString ustr = icu::UnicodeString::fromUTF8(text);

        std::unique_ptr<icu::Transliterator> remover(
            icu::Transliterator::createInstance("NFD; [:M:] Remove; NFC", UTRANS_FORWARD, status)
        );

        if (U_SUCCESS(status)) remover->transliterate(ustr);

        text.clear();
        ustr.toUTF8String(text);
    }

    void WordPiece::handle_chinese_chars_inplace(std::string& text) const {
            std::cerr << "Warning: Method handle_chinese_chars is not implemented" << std::endl;
    }

} // namespace nlp::tokenizer
