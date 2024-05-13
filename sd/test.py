import pickle
import torch
import numpy as np

NEG_INFTY = -1e9
START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
max_sequence_length = 200
malayalam_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', 'ˌ',
                        'ഁ', 'ം', 'ഃ', 'അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'ഌ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ',
                        'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ',
                        'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ',
                        'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ',
                        'ത', 'ഥ', 'ദ', 'ധ', 'ന',
                        'പ', 'ഫ', 'ബ', 'ഭ', 'മ',
                        'യ', 'ര', 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ',
                        '഼', 'ഽ', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'ൃ', 'ൄ', 'െ', 'േ', 'ൈ', 'ൊ', 'ോ', 'ൌ', '്', 'ൎ', 'ൗ', 'ൠ',
                        'ൡ',
                        '൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯', PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      ':', '<', '=', '>', '?', '@',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                      'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                      'Y', 'Z',
                      '[', '\\', ']', '^', '_', '`',
                      'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                      'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z',
                      '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

index_to_malayalam = {k: v for k, v in enumerate(malayalam_vocabulary)}
malayalam_to_index = {v: k for k, v in enumerate(malayalam_vocabulary)}
index_to_english = {k: v for k, v in enumerate(english_vocabulary)}
english_to_index = {v: k for k, v in enumerate(english_vocabulary)}
def create_masks(eng_batch, ml_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    for idx in range(num_sentences):
        eng_sentence_length, ml_sentence_length = len(eng_batch[idx]), len(ml_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        ml_chars_to_padding_mask = np.arange(ml_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, ml_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, ml_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, ml_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask
# Load the trained model
with open('transformer_model.pkl', 'rb') as f:
    transformer = pickle.load(f)



# Switch to evaluation mode
transformer.eval()

# Define the inference function
def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    ml_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_sentence, ml_sentence)
        predictions = transformer(eng_sentence,
                                  ml_sentence,
                                  encoder_self_attention_mask.to(device),
                                  decoder_self_attention_mask.to(device),
                                  decoder_cross_attention_mask.to(device),
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=False)
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = index_to_malayalam[next_token_index]
        ml_sentence = (ml_sentence[0] + next_token, )
        if next_token == END_TOKEN:
            break
    return ml_sentence[0]

# Main function to perform translations
def main():
    while True:
        eng_sentence = input("Enter an English sentence (type 'exit' to quit): ")
        if eng_sentence.lower() == 'exit':
            break
        ml_translation = translate(eng_sentence)
        print("Malayalam Translation:", ml_translation)

if __name__ == "__main__":
    main()

