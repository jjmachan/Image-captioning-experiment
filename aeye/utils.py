from .preprocessing import Lang, PAD_token, SOS_token, EOS_token

def convert_to_text(sample_ids: list, vocab: Lang):
    words = [vocab.index2word[word_id] for word_id in sample_ids]
    return ' '.join(words)
