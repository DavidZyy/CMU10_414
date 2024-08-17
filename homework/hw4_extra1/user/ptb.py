import needle as ndl
# sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

if __name__ == "__main__":
    # device = ndl.cpu()
    device = ndl.cuda()
    base_folder = "/home/zhuyangyang/Course/CMU10_414/homework/hw4/data/lstm/data"
    corpus = ndl.data.Corpus(base_folder, max_lines=100)
    train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
    test_data = ndl.data.batchify(corpus.test, batch_size=16, device=device, dtype="float32")
    model = LanguageModel(30, len(corpus.dictionary), hidden_size=100, num_layers=2, seq_model='rnn', device=device, dtype="float32")
    for i in range(100):
        train_ptb(model, train_data, seq_len=1, n_epochs=1, device=device)
        evaluate_ptb(model, train_data, seq_len=40, device=device)