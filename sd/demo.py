from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import torch
from transformers import CLIPTokenizer
import model_loader, pipeline
import pickle
import os

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)


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

# Load the trained model
with open('transformer_model.pkl', 'rb') as f:
    transformer = pickle.load(f)

# Switch to evaluation mode
transformer.eval()

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


def translate(eng_sentence):
    eng_sentence = (eng_sentence,)
    ml_sentence = ("",)
    for word_counter in range(max_sequence_length):
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_sentence, ml_sentence)
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
        ml_sentence = (ml_sentence[0] + next_token,)
        if next_token == END_TOKEN:
            break
    return ml_sentence[0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_request', methods=['POST'])
def process_request():
    selected_option = request.form['selected_option']
    if selected_option == 'translation':
        eng_sentence = request.form['eng_sentence']
        ml_translation = translate(eng_sentence)
        return render_template('results.html', translation=ml_translation)
    elif selected_option == 'text_to_image':
        text_prompt = request.form['text_prompt']
        try:
            text_to_image_output = pipeline.generate(
                prompt=text_prompt,
                uncond_prompt="",
                input_image=None,
                strength=0.5,
                do_cfg=True,
                cfg_scale=8,
                sampler_name="ddpm",
                n_inference_steps=2,
                seed=42,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            image_path = os.path.join('static', 'generated_image.jpg')
            text_to_image_output_pil = Image.fromarray(np.uint8(text_to_image_output))
            text_to_image_output_pil.save(image_path)
            return render_template('results.html', image_path=image_path)
        except Exception as e:
            return str(e)
    elif selected_option == 'image_to_image':
        image_file = request.files['image_file']
        strength = float(request.form['strength'])

        if image_file.filename == '':
            return 'No image selected', 400

        try:
            input_image = Image.open(image_file.stream)
            image_to_image_output = pipeline.generate(
                prompt="",
                uncond_prompt="",
                input_image=input_image,
                strength=strength,
                do_cfg=True,
                cfg_scale=8,
                sampler_name="ddpm",
                n_inference_steps=4,
                seed=42,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            image_to_image_output_pil = Image.fromarray(np.uint8(image_to_image_output))

            # Save both input and output images
            input_image_path = os.path.join('static', 'input_image.jpg')
            output_image_path = os.path.join('static', 'output_image.jpg')
            input_image.save(input_image_path)
            image_to_image_output_pil.save(output_image_path)

            return render_template('results.html', input_image=input_image_path, output_image=output_image_path)
        except Exception as e:
            return str(e), 500
    elif selected_option == 'text_to_image_image_to_image':
        text_prompt = request.form['text_prompt']
        translation = translate(text_prompt)
        try:
            text_to_image_output = pipeline.generate(
                prompt=text_prompt,
                uncond_prompt="",
                input_image=None,
                strength=0.5,
                do_cfg=True,
                cfg_scale=8,
                sampler_name="ddpm",
                n_inference_steps=2,
                seed=42,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            # Convert text-to-image output to a PIL image
            text_to_image_output_pil = Image.fromarray(np.uint8(text_to_image_output))

            # Use the text-to-image output as input for image-to-image
            image_to_image_output = pipeline.generate(
                prompt="",
                uncond_prompt="",
                input_image=text_to_image_output_pil,
                strength=0.5,  # Adjust strength as needed
                do_cfg=True,
                cfg_scale=8,
                sampler_name="ddpm",
                n_inference_steps=4,
                seed=42,
                models=models,
                device=DEVICE,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            # Convert image-to-image output to a PIL image
            image_to_image_output_pil = Image.fromarray(np.uint8(image_to_image_output))

            # Save the generated images
            text_to_image_output_path = os.path.join('static', 'text_to_image_output.jpg')
            image_to_image_output_path = os.path.join('static', 'image_to_image_output.jpg')
            text_to_image_output_pil.save(text_to_image_output_path)
            image_to_image_output_pil.save(image_to_image_output_path)

            # Render the results template with the generated images
            return render_template('results.html', translation=translation, image_path=None, input_image=text_to_image_output_path, output_image=image_to_image_output_path)
        except Exception as e:
            return str(e)
    else:
        return 'Invalid option selected', 400

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == "__main__":
    app.run(debug=True)
