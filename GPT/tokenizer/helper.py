def get_stats(ids, stats_arg = None):
    stats = {} if stats_arg is None else stats_arg

    for pair in zip(ids, ids[1:]):
        stats[pair] = stats.get(pair, 0) + 1

    return stats
    
def merge(ids, pair, idx):
    # specifically for optimizing RegexTokenizer
    if pair[0] not in ids or pair[1] not in ids:
        return ids
    
    newids = []
    i = 0

    while i < len(ids):
        if i == len(ids) - 1:
            newids.append(ids[i])
            break
        elif ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
                
    return newids  