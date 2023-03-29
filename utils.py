import torch
import sys


def translate_sentence(input_sentence, model, source_vectorization, target_vectorization, sequence_length):
    tokenized_input_sentence = torch.tensor(source_vectorization([input_sentence]).numpy())
    decoded_sentence = "[start]"
    for i in range(sequence_length):
        tokenized_target_sentence = torch.tensor(target_vectorization([decoded_sentence])[:, :-1].numpy())
        output = model(tokenized_input_sentence, tokenized_target_sentence)
        sampled_token_index = output.argmax(2)[0, i].item()
        sampled_token = target_vectorization.get_vocabulary()[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])