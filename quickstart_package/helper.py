import math


def min_feature_size(query_size, alpha):
    return int(math.ceil(alpha * alpha * query_size))

def max_feature_size(query_size, alpha):
    return int(math.floor(query_size * 1.0 / (alpha * alpha)))

def minimum_common_feature_count(query_size, y_size, alpha):
    return int(math.ceil(alpha * math.sqrt(query_size * y_size)))

def cosine_sim(X, Y):
    return len(set(X) & set(Y)) * 1.0 / math.sqrt(len(set(X)) * len(set(Y)))


def min_feature_size(self, query_size, alpha):
        return int(math.ceil(alpha * query_size))

def max_feature_size(query_size, alpha):
    return int(math.floor(query_size / alpha))

def minimum_common_feature_count(query_size, y_size, alpha):
    return int(math.ceil(alpha * (query_size + y_size) * 1.0 / (1 + alpha)))

def jaccard_sim(X, Y):
    return len(set(X) & set(Y)) * 1.0 / len(set(X) | set(Y))