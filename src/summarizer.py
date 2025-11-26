from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

def summarize_text(text, num_sentences=3):
    """
    Summarize the given text using TextRank algorithm.
    Returns a summary with the specified number of sentences.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()

    summary_sentences = summarizer(parser.document, num_sentences)

    return " ".join(str(sentence) for sentence in summary_sentences)


def summarize_cluster(texts, num_sentences=3):
    """
    Combine all articles from a cluster and summarize the cluster.
    """
    combined_text = " ".join(texts)

    return summarize_text(combined_text, num_sentences=num_sentences)
