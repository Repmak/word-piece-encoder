#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>
#include "TokenizerInterface.h"

namespace sentencpp::tokenizer {

    class WordPiece : public TokenizerInterface {
        public:
            explicit WordPiece(const WordPieceConfig& config);

            [[nodiscard]] std::vector<Token> tokenize(std::string_view text) const override;

            [[nodiscard]] size_t get_vocab_size() const override { return vocab_list_->size(); }
            [[nodiscard]] const VocabList& get_vocab_list() const { return *vocab_list_; }

        private:
            WordPieceConfig config_;
            std::unique_ptr<VocabList> vocab_list_;

            // Splits text by whitespace and punctuation.
            [[nodiscard]] static std::vector<std::string_view> split_text(std::string_view text);

            // Encode each word into one or more tokens (using MaxMatch algorithm).
            [[nodiscard]] std::vector<Token> encode_word(std::string_view word) const;

            // Truncation and adding special tokens.
            void post_processing(std::vector<Token>& tokens) const;

            // Normalising user input.
            static void clean_text_inplace(std::string& text);
            static void to_lowercase_inplace(std::string& text);
            static void strip_accents_inplace(std::string& text);
            static void handle_chinese_chars_inplace(std::string& text);
    };

} // namespace sentencpp::tokenizer
