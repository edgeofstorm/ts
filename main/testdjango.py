from typing import List
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java,isJVMStarted
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math, statistics, random


def run(title, input,  morphology, compression_ratio=0.5):

    # ZEMBEREK_PATH = r'C:\Users\haQQi\Desktop\zemberek-full.jar'

    # if not isJVMStarted():
    #     startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % (ZEMBEREK_PATH), convertStrings=True)

    # TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
    # morphology = TurkishMorphology.createWithDefaults()

    compression_ratio = compression_ratio

    title = title
    input = input

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

    if title:
        title_analysis: java.util.ArrayList = (
            morphology.analyzeAndDisambiguate(title).bestAnalysis()
        )
        for word_analysis in title_analysis:
            if not word_analysis.formatLexical()[
                   word_analysis.formatLexical().index(':') + 1:word_analysis.formatLexical().index(']')] == 'Punc':
                if not word_analysis.formatLexical()[
                       word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')] == 'UNK':
                    bow.append(word_analysis.formatLexical()[
                               word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')])
                    morphed_title += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                        '[') + 1:word_analysis.formatLexical().index(':')]
        morphed_title = morphed_title.strip()
        print(morphed_title)

    for i, word in enumerate(sentences):
        print(f'Sentence {i + 1}: {word}')
        sentence_lengths.append(len(str(word).split(" ")))
        sent_analysis: java.util.ArrayList = (
            morphology.analyzeAndDisambiguate(word).bestAnalysis()
        )
        morphed_sentence = ""
        for word_analysis in sent_analysis:
            if not word_analysis.formatLexical()[
                   word_analysis.formatLexical().index(':') + 1:word_analysis.formatLexical().index(']')] == 'Punc':
                if not word_analysis.formatLexical()[
                       word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')] == 'UNK':
                    bow.append(word_analysis.formatLexical()[
                               word_analysis.formatLexical().index('[') + 1:word_analysis.formatLexical().index(':')])
                    morphed_sentence += " " + word_analysis.formatLexical()[word_analysis.formatLexical().index(
                        '[') + 1:word_analysis.formatLexical().index(':')]

        morphed_sentences.append(morphed_sentence.strip())

    if morphed_title:
        morphed_sentences.insert(0, morphed_title)

    print(sentence_lengths)
    print(morphed_sentences)

    stopwordsturkish = set(stopwords.words('turkish'))
    vectorizer = TfidfVectorizer(stop_words=stopwordsturkish, use_idf=True, ngram_range=(1, 2))  # , ngram_range=(1, 3)
    X = vectorizer.fit_transform(morphed_sentences)  # [input]
    feature_vectors = vectorizer.transform(morphed_sentences)
    ts = []
    for i in range(len(morphed_sentences)):
        if i == 0:
            continue
        ts.append(cosine_similarity(feature_vectors[0], feature_vectors[i])[0][0])
    tfidf_scores = []

    for arr in X.toarray():
        arr_len = 0
        total = 0
        for elem in arr:
            if elem != 0:
                total += elem
                arr_len += 1
        tfidf_scores.append(total / arr_len)  # / sentence_lengths[counter]
    print(tfidf_scores)

    maximum = []
    average = []
    minimum = []

    morphed_sentences.pop(0)


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

        def score_fitness(self):
            self.fitness += 0.025 * fitnessSL(self.genes)  # 0.4
            self.fitness += 0.06 * fitnessSP(self.genes)  # 0.4
            self.fitness += 0.05 * fitnessTFIDF(self.genes, tfidf_scores)  # 0.4
            self.fitness += 0.6 * fitnessTS(self.genes, ts)  # 0.4
            return self.fitness

        def mutate(self, mutation_rate):
            # counter for guaranteeing mutation for every 1000 ?
            for i, elem in enumerate(self.genes):
                if random.random() < mutation_rate:
                    while ():
                        rand = random.randint(0, len(self.genes))
                        if self.genes[rand] == 1:
                            self.genes[i] = 0
                            break
                    while ():
                        rand = random.randint(0, len(self.genes))  # +1
                        if self.genes[rand] == 0:
                            self.genes[i] = 1
                            break

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

        def __init__(self,
                     target=morphed_sentences,
                     mutation_rate=0.05,
                     crossover_rate=0.5,
                     population=100,
                     compression_rate=0.5,
                     iteration_count=50,
                     ):
            self.target = target
            self.mutation_rate = mutation_rate
            self.crossover_rate = crossover_rate
            self.population_count = population
            self.population = []
            self.compression_rate = compression_rate
            self.iteration_count = iteration_count

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

        def selection_rank_space(self):
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
                curr = gene.score_fitness()
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

            fittest_ones = []

            for i in range(self.iteration_count):
                print("Generation {}".format(i))
                fittest_ones.append(self.calculate_fitness())
                print(self.fittest().genes)
                if i > 5:
                    break_check = [gene.fitness for gene in fittest_ones[-5:]]
                    if len(set(break_check)) == 1:
                        print(
                            f"algorithm converged in {i} steps (last 5 generation's fittest individuals are the same)")
                        return

                diff_partners = []
                diff = 0
                new_population = []

                for i in range(len(self.population)):

                    first_partner = self.selection_roulette_wheel(self.selection_rank_space())
                    second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                    while first_partner == second_partner:
                        second_partner = self.selection_roulette_wheel(self.selection_rank_space())

                    diff_partners.append(first_partner)
                    diff_partners.append(second_partner)

                    new_child = first_partner.crossover(second_partner)
                    new_child.mutate(self.mutation_rate)
                    new_population.append(new_child)
                    # self.population[i] = new_child

                self.population = new_population

        def populate(self):
            sum_length = round(len(self.target) * self.compression_rate)
            input_length = len(self.target)
            if sum_length < 4 and input_length > 4:
                sum_length = 4
            for i in range(self.population_count):
                random_gene = Gene(sum_length, input_length)
                self.population.append(random_gene)
            print("population created with {} individuals".format(self.population_count))

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

    def fitnessSP(gene):
        sp_score = 0
        for i, elem in enumerate(gene, start=1):
            if elem == 1:
                sp_score += math.sqrt(1 / i)
        fit_sp.append(sp_score)
        return sp_score


    def fitnessTS(gene, ts):
        ts_score = 0
        for i, elem in enumerate(gene):
            if elem == 1:
                ts_score += ts[i]
        fit_ts.append(ts_score)
        return ts_score


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


    def generate_summary(fittest_individual):
        summary = ""
        for i, elem in enumerate(fittest_individual.genes):
            if elem == 1:
                summary += sentences[i] + " "
        return summary.strip()


    genetic = Genetic(morphed_sentences, compression_rate=compression_ratio, population=100, iteration_count=25)
    genetic.populate()
    genetic.iterate()
    print(genetic.fittest().genes)
    
    return generate_summary(genetic.fittest())


