import argparse
import pickle


from train import Model, defaultdict_gen


def generate(model_path, prefix, length):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    words = model.preprocess(prefix)
    for i in range(len(words), length):
        words.append(model.generate(words[-model.n:]))
    text = model.postprocess(words)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate')
    parser.add_argument('--model', dest='model_path', type=str, help='Path to file to load trained model')
    parser.add_argument('--prefix', type=str, default="", help='Beginning of text')
    parser.add_argument('--length', type=int, help='Number of words in text')
    args = parser.parse_args()

    print(generate(args.model_path, args.prefix, args.length))