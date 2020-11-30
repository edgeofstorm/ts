import time
from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java,isJVMStarted
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import math, statistics, random


def tokenize(input):
    TurkishSentenceExtractor: JClass = JClass(
        'zemberek.tokenization.TurkishSentenceExtractor'
    )
    extractor: TurkishSentenceExtractor = TurkishSentenceExtractor.DEFAULT
    sentences = extractor.fromParagraph(input)
    return sentences

def preprocess(input):
    pass

def run(title, input,  morphology, ner_model, compression_ratio=0.5, coeffs={},elitist=True,mutation_rate=0.06,population=100, crossover_rate=1.0, iter_count=25, selection = 'Rank Space'):

    title = title
    input = input
    selection=selection
    compression_ratio=compression_ratio
    coeffs = coeffs
    elitist=elitist
    mutation_rate=mutation_rate
    population=population
    if population > 250:
        population=250
    iter_count=iter_count
    if iter_count > 50:
        iter_count=50
    crossover_rate=crossover_rate

    info=[f'coefficients = {coeffs}']

    sentences = tokenize(input)

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
                       word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')] == 'UNK':
                    # COUNTER AND SENTENCE_LENGTHS[]
                    bow.append(word_analysis.formatLexical()[
                               word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')])
                    morphed_title += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                        '[') + 1:word_analysis.formatLexical().index(':')]
        morphed_title = morphed_title.strip()

    have_title = True if morphed_title else False

    # morph input
    for i, word in enumerate(sentences):
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
                       word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')] == 'UNK':
                    # COUNTER AND SENTENCE_LENGTHS[]
                    bow.append(word_analysis.formatLexical()[
                               word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')])
                    morphed_sentence += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                        '[') + 1:word_analysis.formatLexical().index(':')]

        morphed_sentences.append(morphed_sentence.strip())

    if have_title:
        morphed_sentences.insert(0, morphed_title)

    info.append(f'total nu. of sentences in the doc = {len(sentence_lengths)}')
    info.append(f'total nu. of words in the doc = {sum(sentence_lengths)}')
    

    # stopwordsturkish = set(stopwords.words('turkish'))
    stopwords = ['acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']
    vectorizer = TfidfVectorizer(stop_words=stopwords, use_idf=True, ngram_range=(1, 2))  # , ngram_range=(1, 3)
    X = vectorizer.fit_transform(morphed_sentences)  # [input]
    # for sentence similarity
    feature_vectors = vectorizer.transform(morphed_sentences)
    ts = []
    if have_title:
        for i in range(len(morphed_sentences)):
            if i == 0:
                continue
            ts.append(linear_kernel(feature_vectors[0], feature_vectors[i])[0][0])
    tfidf_scores = []

    for arr in X.toarray():
        arr_len = 0
        total = 0
        for elem in arr:
            if elem != 0:
                total += elem
                arr_len += 1
        tfidf_scores.append(total / arr_len)  # / sentence_lengths[counter]


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
            if random.random() < mutation_rate:
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
            return False

        def crossover(self, parent):  # crossover rate ?
            child = Gene(self.sum_length, self.input_length)
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
            # assert selection_method in ['Tournament', 'Roulette Wheel',
            #                             'Rank Space', 'Random'], 'selection method must be either Rank Space, Tournament, Random or Roulette Wheel'
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

            threshold = random.uniform(fitnesses[len(fitnesses) - 1][1], fitnesses[0][1])
            while True:
                random_pick = random.randint(0, len(fitnesses) - 1)  # index error
                if threshold < fitnesses[random_pick][1]:
                    return self.population[fitnesses[random_pick][0]]
            return 0

        def selection_roulette_wheel(self, probs):
            rand = random.uniform(0, sum(probs.values()))
            temp = [(key, probs[key]) for key in probs]
            random.shuffle(temp)
            curr = 0
            for i in range(len(temp)):
                curr += temp[i][1]
                if curr >= rand:
                    return self.population[temp[i][0]]
            return 0

        def selection_rank_space(self):  # 0 < Pc < 1
            Pc = 0.7
            rank = {}
            fitnesses = self.sort_population_and_remove_duplicates()

            for i, my_tuple in enumerate(fitnesses):
                l = list(my_tuple)
                rank[l[0]] = len(fitnesses) - i
            return rank

        def calculate_fitness(self):
            max = 0
            min = 0
            total = 0
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
            # print("avg fitness -> {} . Fittest individual {} least fittest {}".format(avg, max, min))
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

            fittest_ones = []

            for i in range(self.iteration_count):
                # print("Generation {}".format(i))
                fittest_ones.append(self.calculate_fitness())
                # print(self.fittest().genes)
                if i > 5:
                    break_check = [gene.fitness for gene in fittest_ones[-5:]]
                    if len(set(break_check)) == 1:
                        info.append(
                            f"algorithm converged in {i} steps (last 5 generation's fittest individuals are the same)")
                        return
                
                diff_partners = []
                diff = 0
                new_population = []  # declare outside loop then use .clear()
                count = len(self.population)
                if self.elitist:
                    new_population.append(self.fittest())
                    count -= 1
                #new_population.append(self.fittest())
                for i in range(count):

                    first_partner = self.selection_roulette_wheel(self.selection_rank_space())
                    second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                    while first_partner == second_partner:
                        second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                    diff_partners.append(first_partner)
                    diff_partners.append(second_partner)

                    new_child = first_partner.crossover(second_partner)
                    if new_child.mutate(self.mutation_rate) is True:
                        self.mutation_count += 1
                    new_population.append(new_child)
                    # self.population[i] = new_child

                self.population = new_population
                self.iterated += 1

        def populate(self):
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
            info.append("population created with {} individuals".format(self.population_count))

        def re_initialize(self, fit_count=5):
            population = []
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
            return fitnesses

        def sort_population_and_remove_duplicates(self):
            fitnesses = self.sort_population_by_fitness()
            length = len(fitnesses)
            for i in range(length):
                if i + 1 >= len(fitnesses):
                    break
                while fitnesses[i][1] == fitnesses[i + 1][1]:
                    del fitnesses[i]
                    if i + 1 >= len(fitnesses):
                        break
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
        fit_sl.append(sl_score)
        return sl_score


    def fitnessNER(gene, ner):
        if not have_ner:
            return 0
        ner_score = 0
        ner_rate = (len(ner) - ner.count(0)) / len(ner)
        for i, elem in enumerate(gene):
            if elem == 1 and ner[i] != 0:
                ner_score += ner[i]
        #fix later
        ner_score = ner_rate * ner_score
        fit_ner.append(ner_score)
        return ner_score


    def generate_summary(fittest_individual):
        summary = ""
        for i, elem in enumerate(fittest_individual.genes):
            if elem == 1:
                summary += sentences[i] + " "
        summary.strip()
        return summary


    def parameter_tune(morphed_sentences, compression_rate):
        test = Genetic(morphed_sentences, compression_rate=compression_rate, population=100, iteration_count=2)
        test.populate()
        test.iterate()
        # add user specific coefs
        if have_ner and have_title:
            if coeffs:
                coefs = {"sl": coeffs['sl'] / max(fit_sl), "sp": coeffs['sp'] / max(fit_sp), "ts": coeffs['ts'] / max(fit_ts),
                     "tfidf": coeffs['tfidf'] / max(fit_tfidf), "ner": coeffs['ner'] / max(fit_ner)}
            else:
                coefs = {"sl": 0.225 / max(fit_sl), "sp": 0.275 / max(fit_sp), "ts": 0.275 / max(fit_ts),
                        "tfidf": 0.125 / max(fit_tfidf), "ner": 0.1 / max(fit_ner)}
        elif have_ner and not have_title:
            if coeffs:
                coefs = {"sl": coeffs['sl'] / max(fit_sl), "sp": coeffs['sp'] / max(fit_sp),
                     "tfidf": coeffs['tfidf'] / max(fit_tfidf), "ner": coeffs['ner'] / max(fit_ner)}
            else:
                coefs = {"sl": 0.35 / max(fit_sl), "sp": 0.35 / max(fit_sp),
                        "tfidf": 0.25 / max(fit_tfidf), "ner": 0.05 / max(fit_ner)}
        elif have_title and not have_ner:
            if coeffs:
                coefs = {"sl": coeffs['sl'] / max(fit_sl), "sp": coeffs['sp'] / max(fit_sp), "ts": coeffs['ts'] / max(fit_ts),
                     "tfidf": coeffs['tfidf'] / max(fit_tfidf)}
            else:
                coefs = {"sl": 0.25 / max(fit_sl), "sp": 0.3 / max(fit_sp), "ts": 0.3 / max(fit_ts),
                        "tfidf": 0.15 / max(fit_tfidf)}
        else:
            if coeffs:
                coefs = {"sl": coeffs['sl'] / max(fit_sl), "sp": coeffs['sp'] / max(fit_sp),
                     "tfidf": coeffs['tfidf'] / max(fit_tfidf)}
            else:
                coefs = {"sl": 0.35 / max(fit_sl), "sp": 0.4 / max(fit_sp),
                        "tfidf": 0.25 / max(fit_tfidf)}

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
    genetic = Genetic(morphed_sentences, compression_rate=compression_ratio, population=population, iteration_count=iter_count,
                      elitist=True,
                      coefs=coefs,mutation_rate=mutation_rate,crossover_rate=crossover_rate,selection_method=selection)
    genetic.populate()
    genetic.iterate()
    info.append(f' fittest chromosome = {genetic.fittest().genes} with a score of {genetic.fittest().fitness}')  # + "     " + str(max(genetic.fittest_ones))     Gene.score_fitness(genetic.fittest(), coefs=coefs)
    info.append(f'{genetic.mutation_count} mutations happened over {genetic.population_count * (genetic.iterated)} genes')
    summary = generate_summary(genetic.fittest())
    end = time.time()
    elapsed = end - start
    info.append(f'genetic algorithm took {elapsed} seconds')

    info.insert(3, f'total nu. of sentences in the summary = {genetic.fittest().sum_length}')
    info.insert(4, f"total nu. of words in the summary = {summary.count(' ')}")


    return summary, info


