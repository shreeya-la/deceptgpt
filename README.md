# DeceptGPT

Online services can use deceptive framing to manipulate users into making decisions that may compromise their privacy. Therefore, we use Natural Language Processing (NLP) techniques to analyze and detect manipulative language in online privacy policies of various websites and applications. We used the OPP-115 Corpus, a set of 115 online privacy policies.

Uploaded files
- rename.py: The original file names have non-consecutive numbers (ranging from 20 to 1713) followed by the respective company name. Therefore, to better organize the policies, we wrote this python script to renumber the policies from 1 to 115. For example, the first policy was renamed from "20_www.theatlantic.com.html" to "1_www.theatlantic.com.html."
- extract-sentences.py: The policies are html documents. Therefore, we wrote this python script to extract the sentences from each privacy policy and clean/process the text. The result is a csv file with three rows: document number, sentence number, and sentence. 
