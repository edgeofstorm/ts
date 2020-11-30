
# from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java

# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# from numpy import dot
# from numpy.linalg import norm
# import numpy as np

import math, statistics, random

import matplotlib.pyplot as plt
from matplotlib import rcParams

import time
from pyrouge import Rouge

if __name__ == '__main__':

    def pre_process():
        pass


    ZEMBEREK_PATH = r'C:\Users\haQQi\Desktop\zemberek-full.jar'
    startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH), convertStrings=True)

    TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
    PerceptronNer = JClass('zemberek.ner.PerceptronNer')
    NamedEntity = JClass('zemberek.ner.NamedEntity')
    morphology = TurkishMorphology.createWithDefaults()
    # ner_analysis: java.util.ArrayList = (
    #     morphology.analyzeAndDisambiguate(title).bestAnalysis()
    # )
    ner_model_path: java.nio.file.Paths = (
        java.nio.file.Paths.get("C:/Users/haQQi/Desktop/ts/ts/resources/my-model")
    )
    ner_model: PerceptronNer = (
        PerceptronNer.loadModel(ner_model_path, morphology)
    )


    def run(title, input, comp, morph, ner):

        compression_ratio = comp
        title = title
        input = input
        morphology = morph
        ner_model = ner

        TurkishSentenceExtractor: JClass = JClass(
            'zemberek.tokenization.TurkishSentenceExtractor'
        )
        extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT
        sentences = extractor.fromParagraph(input)

        desired_length = round(len(sentences) * compression_ratio)
        bow = []
        morphed_sentences = []
        sentence_lengths = []
        morphed_title = ''

        ner = []
        for sentence in sentences:
            named_entities: java.util.ArrayList = (
                ner_model.findNamedEntities(sentence).getNamedEntities()
            )
            ner.append(named_entities.size())
        have_ner = False if ner.count(0) == len(ner) else True
        # print(have_ner)
        print(ner)
        # print(statistics.stdev(ner))
        #
        # ner_dev = [1 if i != 0 else 0 for i in ner]
        # print(ner_dev)
        # print(statistics.stdev(ner_dev))
        # print(statistics.pstdev(ner_dev))

        ner_rate = (len(ner) - (len(ner) - ner.count(0))) / len(ner)

        if ner.count(0) == len(ner) - 1 or ner_rate > 1:
            ner_rate = 1.0
        elif ner.count(0) == 0 or ner_rate < 0.1:
            ner_rate = 0.1

        if title:
            # morph title
            title_analysis: java.util.ArrayList = (
                morphology.analyzeAndDisambiguate(title).bestAnalysis()
            )
            for word_analysis in title_analysis:
                if not word_analysis.formatLexical()[
                       word_analysis.formatLexical().index(':') + 1:word_analysis.formatLexical().index(']')] == 'Punc':
                    if not word_analysis.formatLexical()[
                           word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(
                               ':')] == 'UNK':
                        # COUNTER AND SENTENCE_LENGTHS[]
                        bow.append(word_analysis.formatLexical()[
                                   word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(
                                       ':')])
                        morphed_title += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                            '[') + 1:word_analysis.formatLexical().index(':')]
            morphed_title = morphed_title.strip()
            print(morphed_title)

        have_title = True if morphed_title else False

        # morph input
        for i, word in enumerate(sentences):
            print(f'Sentence {i + 1}: {word}')
            sentence_lengths.append(len(str(word).split(" ")))
            # tf[i+1] = word
            sent_analysis: java.util.ArrayList = (
                morphology.analyzeAndDisambiguate(word).bestAnalysis()
            )
            morphed_sentence = ""
            for word_analysis in sent_analysis:
                if not word_analysis.formatLexical()[
                       word_analysis.formatLexical().index(':') + 1:word_analysis.formatLexical().index(']')] == 'Punc':
                    if not word_analysis.formatLexical()[
                           word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(
                               ':')] == 'UNK':
                        # COUNTER AND SENTENCE_LENGTHS[]
                        bow.append(word_analysis.formatLexical()[
                                   word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(
                                       ':')])
                        morphed_sentence += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                            '[') + 1:word_analysis.formatLexical().index(':')]

            morphed_sentences.append(morphed_sentence.strip())

        if have_title:
            morphed_sentences.insert(0, morphed_title)

        print(sentence_lengths)
        mean = sum(sentence_lengths) / len(sentence_lengths)
        stdev = statistics.stdev(sentence_lengths)  # pstdev()
        print("mean -> " + str(mean))
        print("stdev -> " + str(stdev))
        print(morphed_sentences)

        # stopwordsturkish = set(stopwords.words('turkish'))
        stopwords = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok',
                     'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her',
                     'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde',
                     'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya',
                     'yani']
        vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1, 2))  # , ngram_range=(1, 3)
        X = vectorizer.fit_transform(morphed_sentences)  # [input]
        print(X.shape)
        print(vectorizer.get_feature_names())
        # for sentence similarity
        feature_vectors = vectorizer.transform(morphed_sentences)
        ts = []
        # (Note that the tf-idf functionality in sklearn.feature_extraction.text can produce normalized vectors,
        # in which case cosine_similarity is equivalent to linear_kernel, only slower.)
        # cosine similarity -> 0.1974,0.435 s
        # linear kernel -> 0.0149, 0.2 s

        if have_title:
            for i in range(len(morphed_sentences)):
                if i == 0:
                    continue
                ts.append(linear_kernel(feature_vectors[0], feature_vectors[i])[0][0])
                # print(f"CS title similarity with {i} th sentence" + str(
                #     cosine_similarity(feature_vectors[0], feature_vectors[i])[0][0]))
                # print(f"LK title similarity with {i} th sentence" + str(
                #     linear_kernel(feature_vectors[0], feature_vectors[i])[0][0]))
        tfidf_scores = []

        # counter = 0
        # for arr in X.toarray():
        #     tfidf_scores.append(sum(arr) / sentence_lengths[counter])  # index error , setntence_lengths.insert(0,title_len)
        #     counter += 1
        # print(tfidf_scores)

        for arr in X.toarray():
            arr_len = 0
            total = 0
            for elem in arr:
                if elem != 0:
                    total += elem
                    arr_len += 1
            tfidf_scores.append(total / arr_len)  # / sentence_lengths[counter]
        print(tfidf_scores)

        def maxFreq(sentence):
            terms = sentence.split(" ")
            max = sentence.count(terms[0])
            for term in terms:
                curr_max = sentence.count(term)
                if curr_max > max:
                    max = curr_max
            return max

        def docTermCount(term, docs):
            count = 0;
            for sentence in docs:
                if sentence.count(term) > 0:
                    count += 1;
            return count

        weights = {}
        # do one for title
        # morphed_sentences.insert(0,morphed_title)
        for i, sentence in enumerate(morphed_sentences):
            weights[sentence[:10]] = {}
            for term in sentence.split(" "):
                term_freq = sentence.count(term)
                max_freq = maxFreq(sentence)
                n = len(morphed_sentences)
                nt = docTermCount(term, morphed_sentences)
                weight = (term_freq / max_freq) * math.log(n / nt)
                weights[sentence[:10]][term] = weight
        print(weights)

        sentence_weights = {}
        for i in weights:
            sentence_weights[i] = 0
            count = 0
            for elem in weights[i]:
                sentence_weights[i] += weights[i][elem]
                count += 1
            sentence_weights[i] = sentence_weights[i] / count

        print(list(sentence_weights.values()))

        # iter_a = list(weights[morphed_sentences[0][:10]].values())
        # print(list(weights[morphed_sentences[0][:10]].values()))
        # iter_b = list(weights[morphed_sentences[5][:10]].values())
        # iter_a = np.array(iter_a)#.reshape(np.array(iter_a).shape[0],1)
        # iter_b = np.array(iter_b)#.reshape(np.array(iter_b).shape[0],1)
        # print(iter_a.shape, iter_b.shape)
        # #print(np.transpose(np.array(iter_b).shape))
        # cos_sim = iter_a.dot(iter_b) / (norm(iter_a) * norm(iter_b))
        # print(cos_sim)
        #
        def similarity(s1, s2):
            # max = max(len(s1),len(s2))
            # s1tcounter=0
            # s2tcounter=0
            pay = 0
            payda1 = 0
            payda2 = 0
            for term in s1.split(" "):
                if term in s2.split(" "):
                    wt1 = weights[s1[:10]][term]
                    wt2 = weights[s2[:10]][term]
                    pay += wt1 * wt2
                    payda1 += wt1 * wt1
                    payda2 += wt2 * wt2
                    # wt = wt1 * wt2 / (math.sqrt(pow(wt1,2)) * math.sqrt(pow(wt2,2)))
            payda1 = 0
            payda2 = 0
            for term in s1.split(" "):
                payda1 += weights[s1[:10]][term] * weights[s1[:10]][term]
            for term in s2.split(" "):
                payda2 += weights[s2[:10]][term] * weights[s2[:10]][term]
            a = math.sqrt(payda2)
            b = math.sqrt(payda1)
            c = pay
            print(a, b, c)
            wt = pay / (a * b)
            return wt

        # print(similarity(morphed_sentences[0], morphed_sentences[5]))

        maximum = []
        average = []
        minimum = []

        # remove title if present
        if have_title:
            morphed_sentences.pop(0)

        iter_count = round(math.sqrt(len(sentences) * 7))

        class Gene:
            genes = []
            fitness = 0

            def __init__(self,
                         sum_length=10,
                         input_length=50,
                         ):
                self.genes = [1 for i in range(sum_length)]
                self.genes += [0 for i in range(input_length - sum_length)]
                random.shuffle(self.genes)

                self.sum_length = sum_length
                self.input_length = input_length

            def score_fitness(self, coefs):
                self.fitness += coefs['sl'] * fitnessSL(self.genes)  # %25 0.4  0.025
                self.fitness += coefs['sp'] * fitnessSP(self.genes)  # %30 0.4  0.06
                self.fitness += coefs['tfidf'] * fitnessTFIDF(self.genes, tfidf_scores)  # %15 0.4  0.05
                if have_title:  # 'ts' in coefs.keys():
                    self.fitness += coefs['ts'] * fitnessTS(self.genes, ts)  # %30 0.4  0.6
                if have_ner:
                    self.fitness += coefs['ner'] * fitnessNER(self.genes, ner)
                return self.fitness

            def mutate(self, mutation_rate):
                # counter for guaranteeing mutation for every 1000 ?
                if random.random() < mutation_rate:
                    # DAFUq is this this supposed to do ?
                    # for i, elem in enumerate(self.genes):
                    #     if random.random() < mutation_rate:
                    while True:
                        rand = random.randint(0, len(self.genes) - 1)
                        if self.genes[rand] == 1:
                            self.genes[rand] = 0
                            break
                    while True:
                        rand = random.randint(0, len(self.genes) - 1)  # +1
                        if self.genes[rand] == 0:
                            self.genes[rand] = 1
                            return True
                    # self.genes[i] = 1 if self.genes[i] == 0 else 0
                return False

            def crossover(self, parent):  # crossover rate ?
                child = Gene(self.sum_length, self.input_length)
                # midpoint = random.randint(0, self.input_length)
                # for i, elem in enumerate(child.genes):
                #     child.genes[i] = self.genes[i] if i > midpoint else parent.genes[i]
                # return child
                for i in range(len(child.genes)):
                    child.genes[i] = 0

                same_ones = 0
                loc_ones = []
                for i in range(len(self.genes)):
                    if self.genes[i] == 1 or parent.genes[i] == 1:
                        loc_ones.append(i)
                    if self.genes[i] == parent.genes[i] == 1:
                        child.genes[i] = 1
                        same_ones += 1
                        loc_ones.pop()

                # same child ? address this
                # if same_ones == self.sum_length:

                if same_ones < self.sum_length:
                    for i in range(self.sum_length - same_ones):
                        random_index = random.choice(loc_ones)
                        child.genes[random_index] = 1
                        loc_ones.remove(random_index)  # del loc_ones[random_index]

                return child

            def __eq__(self, other):
                for i in range(len(self.genes)):
                    if self.genes[i] != other.genes[i]:
                        return False
                return True
                # return self.genes == other.genes

            # def __add__(self, other):
            #     return self.fitness + other

            # def lt, gt dunder

            def __str__(self):
                return str(self.genes)

        def gene_set(a):
            # return set(tuple(row.genes) for row in a)
            genes = [tuple(elem.genes) for elem in a]
            return set(tuple(genes))

        class Genetic:

            mating_pool = []
            mutation_count = 0
            iterated = 0

            # fittest_ones=[]

            def __init__(self,
                         target=morphed_sentences,
                         mutation_rate=0.06,
                         crossover_rate=0.5,
                         population=100,
                         compression_rate=0.5,
                         iteration_count=50,
                         elitist=True,
                         selection_method='rank_space',
                         coefs={"sl": 0.025, "sp": 0.06,
                                "ts": 0.6, "tfidf": 0.05, "ner": 0.05}
                         ):
                self.target = target
                self.mutation_rate = mutation_rate
                self.crossover_rate = crossover_rate
                self.population_count = population
                self.population = []
                self.compression_rate = compression_rate
                self.iteration_count = iteration_count
                self.coefs = coefs
                self.elitist = elitist
                assert selection_method in ['tournament', 'roulette_wheel',
                                            'rank_space'], 'selection method must be either rank_space, tournament or roulette_wheel'
                self.selection_method = selection_method

            def mutation(self):
                pass

            def crossover(self):
                pass

            def selection_tournament(self, fitnesses):
                rand1 = random.randint(0, len(fitnesses))
                rand2 = random.randint(0, len(fitnesses))

            def selection_random(self, fitnesses):
                sum = 0
                probs = []
                for i in fitnesses:
                    sum += i[1]
                for i, elem in enumerate(fitnesses):
                    temp = [elem[0], elem[1] / sum]
                    probs.append(temp)

                # threshold = random.uniform(0,0.1)
                # while True:
                #     random_pick = random.randint(0, len(probs)-1)
                #     if threshold < probs[random_pick][1]:
                #         selected = self.population[probs[random_pick][0]]
                #         return selected
                # return 0

                threshold = random.uniform(fitnesses[len(fitnesses) - 1][1], fitnesses[0][1])
                while True:
                    random_pick = random.randint(0, len(fitnesses) - 1)  # index error
                    if threshold < fitnesses[random_pick][1]:
                        return self.population[fitnesses[random_pick][0]]
                return 0

            def selection_roulette_wheel(self, probs):
                rand = random.uniform(0, sum(probs.values()))
                temp = [(key, probs[key]) for key in probs]
                # print(temp)
                random.shuffle(temp)
                # (temp)
                curr = 0
                for i in range(len(temp)):
                    curr += temp[i][1]
                    if curr >= rand:
                        return self.population[temp[i][0]]
                return 0

                # for key in probs:
                #     curr += probs[key]
                #     if curr >= rand:
                #         return self.population[key] #(key,probs[key])
                # return 0

            def selection_rank_space(self):  # 0 < Pc < 1
                Pc = 0.7
                # probabilities = {}
                rank = {}
                fitnesses = self.sort_population_and_remove_duplicates()
                # fitnesses = {}
                # for i, gene in enumerate(self.population):
                #     fitnesses[i] = gene.fitness
                #
                # fitnesses = sorted(fitnesses.items(), key=lambda x: x[1], reverse=True)
                # # maybe do a set to avoid duplicate genes?
                # length = len(fitnesses)
                # for i in range(length):
                #     if i + 1 >= len(fitnesses):
                #         break
                #     while fitnesses[i][1] == fitnesses[i + 1][1]:
                #         del fitnesses[i]
                #         # i+=1
                #         if i + 1 >= len(fitnesses):
                #             break
                #     # if fitnesses[i][1] == fitnesses[i+1][1]:
                #     #     del fitnesses[i+1]
                #     # i= i-1
                #
                # print(fitnesses)
                # a = self.selection_random(fitnesses)
                # print(a)
                # for i,my_tuple in enumerate(fitnesses):
                #     l = list(my_tuple)
                #     if i == 0:
                #         l[1] = Pc
                #         continue
                #     if i == len(fitnesses) -1 :
                #         l[1] = math.pow(1-Pc, i)
                #         pass
                #     l[1] = math.pow(1-Pc, i) * Pc
                #     new_tuple = tuple(l)
                #     probabilities[i] = new_tuple[1]
                #
                # print(probabilities)

                for i, my_tuple in enumerate(fitnesses):
                    l = list(my_tuple)
                    rank[l[0]] = len(fitnesses) - i

                # print(rank)
                # print(len(rank))
                return rank

            def calculate_fitness(self):
                max = 0
                min = 0
                total = 0
                # min=self.population[0].score_fitness()
                # for elitist, will always be at index 0 so skip it to avoid duplicate fitness scoring
                for i, gene in enumerate(self.population):
                    if gene.fitness != 0:
                        curr = gene.fitness
                    else:
                        curr = gene.score_fitness(self.coefs)
                    if i == 0:
                        max = curr
                        min = curr
                        total += curr
                        continue

                    max = curr if curr > max else max
                    min = curr if curr < min else min
                    total += curr

                avg = total / len(self.population)
                print("avg fitness -> {} . Fittest individual {} least fittest {}".format(avg, max, min))
                if min > avg:
                    print('dafuq')
                maximum.append(max)
                average.append(avg)
                minimum.append(min)

                max = 0
                for i, gene in enumerate(self.population):
                    if i == 0:
                        fittest = self.population[i]
                        continue
                    if self.population[i].fitness > fittest.fitness:
                        fittest = self.population[i]

                return fittest

            def iterate(self):
                # mating pool opt.1
                # for i, gene in enumerate(self.population):
                #     fitness = round(self.population[i].fitness * 100)
                #     for j in range(fitness):
                #         self.mating_pool.append(self.population[i])

                # mating pool opt.2
                # mating_pool=[]
                # rankings = self.selection_rank_space()
                # for key in rankings:
                #     for i in range(rankings[key]):
                #         mating_pool.append(self.population[key])
                # random.shuffle(mating_pool)

                # mating pool opt.3
                # for i in range(len(self.population)):
                #     self.mating_pool.append(self.selection_roulette_wheel(self.selection_rank_space()))
                fittest_ones = []
                # print(len(self.mating_pool))
                for i in range(self.iteration_count):
                    if i == 24:
                        print("debug mode on")
                    print("Generation {}".format(i))
                    fittest_ones.append(self.calculate_fitness())
                    print(self.fittest().genes)
                    if i > 5:
                        break_check = [gene.fitness for gene in fittest_ones[-5:]]
                        # check if the genes are the same ?
                        if len(set(break_check)) == 1:
                            print(
                                f"algorithm converged in {i} steps (last 5 generation's fittest individuals are the same)")
                            # ? mutation rate
                            # self.iteration_count -= self.iterated
                            # fittest_ones.clear()
                            # self.re_initialize()
                            # self.iterate()
                            return
                    diff_partners = []
                    diff = 0
                    new_population = []  # declare outside loop then use .clear()
                    # new_population.append(self.fittest())
                    # random.shuffle(self.population)
                    # new generation
                    for i in range(len(self.population)):
                        # elitist
                        # if i == 0:
                        #     new_population.append(self.fittest())
                        #     continue

                        # a = random.randint(0, len(self.mating_pool) - 1)  # IndexError
                        # b = random.randint(0, len(self.mating_pool) - 1)
                        #
                        # first_partner = self.mating_pool[a]
                        # second_partner = self.mating_pool[b]

                        first_partner = self.selection_roulette_wheel(self.selection_rank_space())
                        second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                        # maybe not do this ?
                        while first_partner == second_partner:
                            # second_partner = self.mating_pool[random.randint(0, len(self.mating_pool) - 1)]
                            second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                        # diff_partners.append((first_partner,second_partner))
                        diff_partners.append(first_partner)
                        diff_partners.append(second_partner)

                        new_child = first_partner.crossover(second_partner)
                        if new_child.mutate(self.mutation_rate) is True:
                            self.mutation_count += 1
                        new_population.append(new_child)
                        # self.population[i] = new_child

                    self.population = new_population
                    self.iterated += 1
                # print("new pop")
                # print(len(self.population))
                # print("different parents -> {}".format(len(diff_partners)))
                # print("different parents -> {}".format(len(gene_set(diff_partners))))

            def populate(self):
                # this is for because round(2.5) returns 2 instead of 3.
                frac, whole = math.modf(len(self.target) * self.compression_rate)
                if frac == 0.5:
                    whole += 1
                sum_length = int(whole)  # round(len(self.target) * self.compression_rate)
                input_length = len(self.target)
                if sum_length < 4 and input_length > 4:
                    sum_length = 4
                for i in range(self.population_count):
                    random_gene = Gene(sum_length, input_length)
                    self.population.append(random_gene)
                print("population created with {} individuals".format(self.population_count))

            def re_initialize(self, fit_count=5):
                # when alg converges take top 5 of last gen and re-run
                population = []
                # fit_ones = self.population[i] for i in self.sort_population_and_remove_duplicates()[:5]]
                fit_ones = []
                for i in range(fit_count):
                    fit_ones.append(self.population[self.sort_population_and_remove_duplicates()[i][0]])
                population += fit_ones

                frac, whole = math.modf(len(self.target) * self.compression_rate)
                if frac == 0.5:
                    whole += 1
                sum_length = int(whole)  # round(len(self.target) * self.compression_rate)
                input_length = len(self.target)
                for i in range(self.population_count - fit_count):
                    random_gene = Gene(sum_length, input_length)
                    population.append(random_gene)
                print("population re-initialized with {} individuals with {} fittest  from last gen".format(
                    self.population_count, fit_count))

            def sort_population_by_fitness(self):
                fitnesses = {}
                for i, gene in enumerate(self.population):
                    fitnesses[i] = gene.fitness
                fitnesses = sorted(fitnesses.items(), key=lambda x: x[1], reverse=True)
                # print(fitnesses)
                return fitnesses

            def sort_population_and_remove_duplicates(self):
                fitnesses = self.sort_population_by_fitness()
                length = len(fitnesses)
                for i in range(length):
                    if i + 1 >= len(fitnesses):
                        break
                    while fitnesses[i][1] == fitnesses[i + 1][1]:
                        del fitnesses[i]
                        # i+=1
                        if i + 1 >= len(fitnesses):
                            break
                    # if fitnesses[i][1] == fitnesses[i+1][1]:
                    #     del fitnesses[i+1]
                    # i= i-1

                # print(fitnesses)
                return fitnesses

            def fittest(self):
                return self.population[self.sort_population_by_fitness()[0][0]]

        fit_sp = []
        fit_ts = []
        fit_sl = []
        fit_tfidf = []
        fit_ner = []

        # sentence position score of a gene
        def fitnessSP(gene):
            sp_score = 0
            for i, elem in enumerate(gene, start=1):
                if elem == 1:
                    if i == len(gene):
                        sp_score += math.sqrt(1 / 1.7)  # giving the importance last sentence needs
                        continue
                    sp_score += math.sqrt(1 / i)
            # print('SP score of a gene -> ' + str(sp_score))
            # for start from end
            # if 1
            # score += sqrt of 1 / i*i
            fit_sp.append(sp_score)
            return sp_score

        # title similarity score of a gene
        def fitnessTS(gene, ts):
            if not have_title:
                return 0
            ts_score = 0
            for i, elem in enumerate(gene):
                if elem == 1:
                    ts_score += ts[i]
            # print('TS score of a gene -> ' + str(ts_score))
            fit_ts.append(ts_score)
            return ts_score

        # cohesion and coverage

        def fitnessTFIDF(gene, tfidf):
            selected_sentences = []
            score = 0

            for i, elem in enumerate(gene):
                if elem == 1:
                    selected_sentences.append(i)

            for i in selected_sentences:
                score += tfidf[i]
            # print('tfdif score of a gene -> '+str(score))
            fit_tfidf.append(score)
            return score

        def fitnessSL(gene):
            sl_score = 0
            chosen = []

            for i, elem in enumerate(gene):
                if elem == 1:
                    chosen.append(morphed_sentences[i])

            lengths = [len(chosen[i].split(" ")) for i in range(len(chosen))]
            stdev = statistics.pstdev(lengths) if len(lengths) > 0 else 1
            mean = statistics.mean(lengths) if len(lengths) > 0 else 0
            stdev = 1 if stdev == 0 else stdev
            for i in lengths:
                expr = math.exp(((i + mean) * -1) / stdev)
                sl_score += (1 - expr) / (1 + expr)
            # print('SL score of a gene -> ' + str(sl_score))
            fit_sl.append(sl_score)
            return sl_score

        def fitnessNER(gene, ner):
            if not have_ner:
                return 0
            ner_score = 0
            ner_rate = (len(ner) - ner.count(0)) / len(ner)
            # print(ner_rate)
            for i, elem in enumerate(gene):
                if elem == 1 and ner[i] != 0:
                    ner_score += ner[i]

            # fix this later
            ner_score = ner_rate * ner_score
            fit_ner.append(ner_score)
            return ner_score

        def generate_summary(fittest_individual):
            summary = ""
            for i, elem in enumerate(fittest_individual.genes):
                if elem == 1:
                    summary += sentences[i] + " "
            # print(len(summary))
            # print(str(summary))
            summary.strip()
            print(summary)
            return summary

        def parameter_tune(morphed_sentences, compression_rate):
            test = Genetic(morphed_sentences, compression_rate=compression_rate, population=100, iteration_count=2)
            test.populate()
            test.iterate()

            print(f'SL score min={min(fit_sl)} max={max(fit_sl)}')
            print(f'SP score min={min(fit_sp)} max={max(fit_sp)}')
            print(f'TS score min={min(fit_ts)} max={max(fit_ts)}' if have_title else f'TS score min=0 max=0')
            print(f'NER score min={min(fit_ner)} max={max(fit_ner)}' if have_ner else f'NER score min=0 max=0')
            print(f'tfidf score min={min(fit_tfidf)} max={max(fit_tfidf)}')
            # genetic.__del__

            if have_ner and have_title:
                coefs = {"sl": 0.225 / max(fit_sl), "sp": 0.275 / max(fit_sp), "ts": 0.275 / max(fit_ts),
                         "tfidf": 0.125 / max(fit_tfidf), "ner": 0.1 / max(fit_ner)}
            elif have_ner and not have_title:
                coefs = {"sl": 0.35 / max(fit_sl), "sp": 0.35 / max(fit_sp),
                         "tfidf": 0.25 / max(fit_tfidf), "ner": 0.05 / max(fit_ner)}
            elif have_title and not have_ner:
                coefs = {"sl": 0.25 / max(fit_sl), "sp": 0.3 / max(fit_sp), "ts": 0.3 / max(fit_ts),
                         "tfidf": 0.15 / max(fit_tfidf)}
            else:
                coefs = {"sl": 0.35 / max(fit_sl), "sp": 0.4 / max(fit_sp),
                         "tfidf": 0.25 / max(fit_tfidf)}

            # coefs = {"sl": 0.25 / max(fit_sl), "sp": 0.30 / max(fit_sp), "ts": 0.30 / max(fit_ts),
            #          "tfidf": 0.15 / max(fit_tfidf)} if have_title else {"sl": 0.30 / max(fit_sl), "sp": 0.40 / max(fit_sp),
            #                                                              "tfidf": 0.30 / max(fit_tfidf)}

            print(coefs)

            # CLEAR
            fit_sl.clear()
            fit_sp.clear()
            fit_tfidf.clear()
            fit_ts.clear()
            fit_ner.clear()
            maximum.clear()
            average.clear()
            minimum.clear()

            return coefs

        start = time.time()
        coefs = parameter_tune(compression_rate=compression_ratio, morphed_sentences=morphed_sentences)
        genetic = Genetic(morphed_sentences, compression_rate=compression_ratio, population=100, iteration_count=25,
                          elitist=True,
                          coefs=coefs)
        genetic.populate()
        genetic.iterate()
        print(
            genetic.fittest().genes)  # + "     " + str(max(genetic.fittest_ones))     Gene.score_fitness(genetic.fittest(), coefs=coefs)
        print(
            f'{genetic.mutation_count} mutations happened over {genetic.population_count * (genetic.iterated)} genes')
        generate_summary(genetic.fittest())
        end = time.time()
        elapsed = end - start
        print(f'genetic algorithm took {elapsed} seconds')

        # print(
        #     f'SL score min={min(fit_sl)} max={max(fit_sl)} fittest={fitnessSL(genetic.fittest().genes)} * {coefs["sl"]} = {fitnessSL(genetic.fittest().genes) * coefs["sl"]}')
        # print(
        #     f'SP score min={min(fit_sp)} max={max(fit_sp)} fittest={fitnessSP(genetic.fittest().genes)} * {coefs["sp"]} = {fitnessSP(genetic.fittest().genes) * coefs["sp"]}')
        # print(
        #     f'TS score min={min(fit_ts)} max={max(fit_ts)} fittest={fitnessTS(genetic.fittest().genes, ts)} * {coefs["ts"]} = {fitnessTS(genetic.fittest().genes, ts) * coefs["ts"]}' if have_title else f'TS score min=0 max=0')
        # print(
        #     f'NER score min={min(fit_ner)} max={max(fit_ner)} fittest={fitnessNER(genetic.fittest().genes, ner)} * {coefs["ner"]} = {fitnessNER(genetic.fittest().genes, ner) * coefs["ner"]} ' if have_ner else f'NER score min=0 max=0')
        # print(
        #     f'tfidf score min={min(fit_tfidf)} max={max(fit_tfidf)} fittest={fitnessTFIDF(genetic.fittest().genes, tfidf_scores)} * {coefs["sl"]} = {fitnessTFIDF(genetic.fittest().genes, tfidf_scores) * coefs["tfidf"]}')
        #
        # rcParams['figure.figsize'] = 8, 6
        # rcParams['axes.xmargin'] = 0
        # plt.plot(maximum, label='maximum')
        # plt.plot(average, label='average')
        # plt.plot(minimum, label='minimum')
        # plt.legend(loc=4)
        # plt.grid(True)  # , color='k', linestyle=":"
        # genetic.elitist = False
        # plt.title(
        #     f"Convergence Graph\n Input Length={len(sentences)} Compression Rate=%{int(100 * (1 - genetic.compression_rate))} Population Count={genetic.population_count}\n Mutation Rate={genetic.mutation_rate} Elitist={genetic.elitist}\n")
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness Score')
        # # plt.ylim(minimum[0], round(maximum[len(maximum)-1]))
        # # plt.ylim(4.8, 6)
        # # plt.yticks([4.8 + i / 10 for i in range(13)])
        # plt.xticks([i for i in range(len(maximum))])
        # # plt.style.use()
        #
        # plt.show()
        # print(genetic.selection_rank_space())

        # for i in range(iter_count):
        #     print("generation {}".format(i))
        #     if i == iter_count-1:
        #         fittest_individual = genetic.calculate_fitness()
        #         break
        #     genetic.calculate_fitness()
        #     #print(generate_summary(genetic.calculate_fitness()))
        #     genetic.iterate()
        #
        # summary = generate_summary(fittest_individual)
        # print(summary)
        # print(len(summary))

        return genetic.fittest().genes


    TurkishSentenceExtractor: JClass = JClass(
        'zemberek.tokenization.TurkishSentenceExtractor'
    )
    extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT


    def evaluate_precision(golden_summaries, generated_summaries):
        r = Rouge()

        docs = []
        golden = []
        generated = []
        with open(r"C:\Users\haQQi\Desktop\docs.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        with open(r"C:\Users\haQQi\Desktop\golden.txt", "r", encoding="utf-8") as t:
            l1 = []
            for ele in t:
                line = ele.split('\n')
                a = ele[ele.index('[') + 1:ele.index(']')]
                temp = [int(i) for i in a.split(',')]
                l1.append(temp)
            # for line in t:
            #     line = line.rstrip('\n')
            #     golden.append(line)
            # golden = t.readlines()
        with open(r"C:\Users\haQQi\Desktop\genesf.txt", "r", encoding="utf-8") as s:
            l2 = []
            for ele in s:
                line = ele.split('\n')
                a = ele[ele.index('[') + 1:ele.index(']')]
                temp = [int(i) for i in a.split(',')]
                l2.append(temp)
            # for line in s:
            #     line = line.rstrip('\n')
            #     generated.append(line)
            # generated = s.readlines()

        golden = l1
        generated = l2

        dataset_prec_scores = []
        dataset_recall_scores = []
        dataset_f_score = []
        for i, doc in enumerate(docs):
            prec = 0
            matched_unit = 0
            recovered_unit = 0
            system_generated_summary = ""
            manual_summmary = ""
            arraylist = extractor.fromParagraph(doc)
            sentences = []
            for sentence in arraylist:
                sentences.append(sentence)

            for k, gene in enumerate(generated[i]):
                if gene == 1:
                    system_generated_summary += sentences[k]

            for j, gene in enumerate(golden[i]):
                if gene == 1:
                    manual_summmary += sentences[k]

            # for k,gene in enumerate(golden[i]):
            #     if gene == generated[i][k] == 1:
            #         matched_unit += len(sentences[k].split(" "))
            # for j,gene in enumerate(generated[i]):
            #     if gene == 1:
            #         recovered_unit += len(sentences[k])
            # prec = matched_unit/recovered_unit
            # doc_prec_scores.append(prec)

            [precision, recall, f_score] = r.rouge_l([system_generated_summary], [manual_summmary])

            print("Precision is :" + str(precision) + "\nRecall is :" + str(recall) + "\nF Score is :" + str(f_score))
            dataset_prec_scores.append(precision)
            dataset_recall_scores.append(recall)
            dataset_f_score.append(f_score)

        print(
            f' avg precision -> {sum(dataset_prec_scores) / len(dataset_prec_scores)} \n avg recall -> {sum(dataset_recall_scores) / len(dataset_recall_scores)} \n avg f_score -> {sum(dataset_f_score) / len(dataset_f_score)}')

        return 0


    def generate_summaries_from_dataset(dataset):
        docs = []
        titles = []
        generated_summaries = []
        with open(r"C:\Users\haQQi\Desktop\docs.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        with open(r"C:\Users\haQQi\Desktop\titles.txt", "r", encoding="utf-8") as t:
            titles = t.readlines()
        for i, doc in enumerate(docs):
            generated_summaries.append(run(title=titles[i], input=doc, comp=0.25, morph=morphology, ner=ner_model))
        print(generated_summaries)
        with open(r"C:\Users\haQQi\Desktop\genes.txt", "w+", encoding="utf-8") as s:
            for generated_summary in generated_summaries:
                s.write(str(generated_summary))
                s.write("\n")


    # generate_summaries_from_dataset(0)
    evaluate_precision(0, 0)

    shutdownJVM()
