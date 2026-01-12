#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace nlp::encoder {

    enum class TokenRole { None, Padding, Unknown, Classification, Separator, Mask };

    struct VocabConfig {
        std::optional<std::string> padding = "[PAD]";
        std::optional<std::string> unknown = "[UNK]";
        std::optional<std::string> classification = "[CLS]";
        std::optional<std::string> separator = "[SEP]";
        std::optional<std::string> mask = "[MASK]";  // Not technically needed. Only used during training to force the model to use context clues to guess what word is behind the mask.

        [[nodiscard]] TokenRole get_special_role(std::string_view token_text) const {
            if (padding && token_text == *padding) return TokenRole::Padding;
            if (unknown && token_text == *unknown) return TokenRole::Unknown;
            if (classification && token_text == *classification) return TokenRole::Classification;
            if (separator && token_text == *separator) return TokenRole::Separator;
            if (mask && token_text == *mask) return TokenRole::Mask;
            return TokenRole::None;
        }
    };

    struct SpecialTokenIds {
        std::optional<uint32_t> padding;
        std::optional<uint32_t> unknown;
        std::optional<uint32_t> classification;
        std::optional<uint32_t> separator;
        std::optional<uint32_t> mask;
    };

    class VocabList {
        public:
            VocabList() = default;
            VocabConfig config;

            // Explicitly set an ID to a token.
            bool set_token(const std::string& token, uint32_t id);

            // Getters.
            [[nodiscard]] const std::unordered_map<std::string, uint32_t>& get_string_to_id_map() const { return string_to_id_map_; }
            [[nodiscard]] const SpecialTokenIds& get_special_ids() const { return special_ids_; }

            [[nodiscard]] std::optional<uint32_t> token_to_id(const std::string& token) const;
            [[nodiscard]] std::optional<std::string> id_to_token(uint32_t id) const;
            [[nodiscard]] size_t size() const { return id_to_string_map_.size(); }

        private:
            std::unordered_map<std::string, uint32_t> string_to_id_map_;
            std::vector<std::string> id_to_string_map_;
            SpecialTokenIds special_ids_;
    };

} // namespace nlp::encoder
