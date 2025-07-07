def chunk_by_length(content_list, max_words=1024, min_words=30):
    chunks = []
    current_chunk = []
    current_len = 0

    for para in content_list:
        para = para.strip()
        word_count = len(para.split())

        if word_count == 0:
            continue

        if current_len + word_count <= max_words:
            current_chunk.append(para)
            current_len += word_count
        else:
            if current_len >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_len = word_count
            else:
                current_chunk.append(para)
                current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
