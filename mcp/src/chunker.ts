/**
 * Simple paragraph-aware text chunker.
 * Splits on blank lines, then merges short chunks to approach targetChars.
 */

export interface Chunk {
  text: string;
  /** approximate character offset in original text */
  offset: number;
}

export function chunk(text: string, targetChars = 800, overlapChars = 100): Chunk[] {
  const paragraphs = text.split(/\n{2,}/);
  const chunks: Chunk[] = [];
  let buffer = '';
  let bufferOffset = 0;
  let currentOffset = 0;

  for (const para of paragraphs) {
    const trimmed = para.trim();
    if (!trimmed) {
      currentOffset += para.length + 2;
      continue;
    }

    if (buffer && buffer.length + trimmed.length + 1 > targetChars) {
      chunks.push({ text: buffer.trim(), offset: bufferOffset });
      // Overlap: carry last overlapChars of previous chunk
      const overlap = buffer.slice(-overlapChars);
      buffer = overlap + '\n' + trimmed;
      bufferOffset = currentOffset - overlap.length;
    } else {
      if (!buffer) bufferOffset = currentOffset;
      buffer = buffer ? buffer + '\n\n' + trimmed : trimmed;
    }

    currentOffset += para.length + 2;
  }

  if (buffer.trim()) {
    chunks.push({ text: buffer.trim(), offset: bufferOffset });
  }

  return chunks.length ? chunks : [{ text: text.trim(), offset: 0 }];
}
