#pragma once

#include <string>
#include <vector>
#include <optional>
#include <iostream>

#include "TokenizerInterface.h"

namespace sentencpp::tokenizer {

    class BPE : public TokenizerInterface {
        public:
            BPE(
                const std::string& config_path,
                const std::string& vocab_key,
                const std::string& merge_rules_key,
                bool clean_text,
                bool to_lowercase,
                bool strip_accents,
                bool handle_chinese_chars,
                std::size_t max_input_chars_per_word,
                std::size_t max_length,
                std::string padding_token="[PAD]",
                std::string unknown_token="[UNK]",
                std::string classification_token="[CLS]",
                std::string separator_token="[SEP]",
                std::string mask_token="[MASK]"
            );

            [[nodiscard]] std::vector<Token> tokenize(std::string_view text) const override;

            [[nodiscard]] size_t get_vocab_size() const override { return vocab_list_->size(); }
            [[nodiscard]] const VocabList& get_vocab_list() const { return *vocab_list_; }

        private:
            std::unique_ptr<VocabList> vocab_list_;
            std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges_;
            bool clean_text;
            bool to_lowercase;
            bool strip_accents;
            bool handle_chinese_chars;
            std::size_t max_input_chars_per_word;
            std::size_t max_length;

            std::vector<std::string> bpe_merge_word(std::string_view word) const;
    };

} // namespace sentencpp::tokenizer
