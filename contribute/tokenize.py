from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java,isJVMStarted


def run(document):

    ZEMBEREK_PATH = staticfiles_storage.path('zemberek-full.jar')

    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH), convertStrings=True)

    TurkishSentenceExtractor: JClass = JClass(
        'zemberek.tokenization.TurkishSentenceExtractor'
    )
    extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT
    sentences = extractor.fromParagraph(document)
    val = []#List[str]
    for sentence in sentences:
        val.append(sentence.strip())
    
    return val


