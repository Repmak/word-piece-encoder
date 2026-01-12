#include <iostream>
#include <iomanip>
#include <optional>
#include "./encoder/include/WordPiece.h"

int main() {
    try {
        nlp::encoder::WordPiece encoder(
            std::string(PROJECT_ROOT_PATH) + "/hf_model/tokenizer.json",
            "/model/vocab",
            true,
            true,
            true,
            true,
            128
        );

        const auto& vocab = encoder.get_vocab_list();
        const auto& string_map = vocab.get_string_to_id_map();
        const auto& special_ids = vocab.get_special_ids();

        // std::cout << std::left << std::setw(20) << "Token" << " | " << "ID" << std::endl;
        // std::cout << std::string(30, '-') << std::endl;
        // for (const auto& [token, id] : string_map) {
        //     std::cout << std::left << std::setw(20) << token << " | " << id << std::endl;
        // }
        //
        // std::cout << "\n--- Special Token IDs ---" << std::endl;
        // auto check_and_print = [&](const std::string& label, const std::optional<uint32_t>& id) {
        //     if (id) std::cout << std::left << std::setw(15) << label << " : " << *id << std::endl;
        //     else std::cout << label << " is empty" << std::endl;
        // };
        //
        // check_and_print("Padding", special_ids.padding);
        // check_and_print("Unknown", special_ids.unknown);
        // check_and_print("Classification", special_ids.classification);
        // check_and_print("Separator", special_ids.separator);
        // check_and_print("Mask", special_ids.mask);

        auto tokens = encoder.encode("Thé quick Browñ fox   jumps over \n the lázy dog");

        std::cout << "\n--- Tokenization Results (" << tokens.size() << " tokens) ---\n";
        std::cout << std::left
                  << std::setw(6)  << "Index"
                  << std::setw(10) << "ID"
                  << std::setw(18) << "Token"
                  << "Role" << "\n";
        std::cout << std::string(45, '-') << "\n";

        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string role_str;
            switch (tokens[i].type) {
                case nlp::encoder::TokenRole::Classification:   role_str = "[CLS]"; break;
                case nlp::encoder::TokenRole::Separator:   role_str = "[SEP]"; break;
                case nlp::encoder::TokenRole::Padding:   role_str = "[PAD]"; break;
                case nlp::encoder::TokenRole::Unknown:   role_str = "[UNK]"; break;
                default:                                      role_str = "None";  break;
            }

            std::cout << std::left
                      << std::setw(6)  << i
                      << std::setw(10) << tokens[i].id
                      << std::setw(18) << tokens[i].text
                      << role_str << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
