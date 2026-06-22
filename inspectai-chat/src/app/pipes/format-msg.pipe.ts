import { Pipe, PipeTransform } from '@angular/core';

@Pipe({ name: 'formatMsg', standalone: true, pure: true })
export class FormatMsgPipe implements PipeTransform {

  transform(text: string): string {
    if (!text) return '';

    // Strip __CHART_AVAILABLE__ signal lines (rendered separately via chartSignal)
    let t = text.replace(/📊__CHART_AVAILABLE__[^\n]*/g, '');

    // Pull out fenced code blocks first so later replacements don't touch their content.
    // Each block is replaced by a unique placeholder, restored at the end.
    const codeBlocks: string[] = [];
    t = t.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(this.renderBlock(lang ?? '', code.trimEnd()));
      return `\x00CODE${idx}\x00`;
    });

    // Inline code (single backtick)
    t = t.replace(/`([^`\n]+)`/g, '<code class="inline-code">$1</code>');

    // Bold
    t = t.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Newlines → <br>
    t = t.replace(/\n/g, '<br>');

    // Restore code blocks
    codeBlocks.forEach((block, i) => {
      t = t.replace(`\x00CODE${i}\x00`, block);
    });

    return t.trim();
  }

  private renderBlock(lang: string, code: string): string {
    const isSql = lang.toLowerCase() === 'sql';
    const body  = isSql ? this.highlightSql(code) : this.escapeHtml(code);
    const label = lang ? lang.toUpperCase() : 'CODE';
    return (
      `<div class="msg-code-block${isSql ? ' msg-code-sql' : ''}">` +
        `<div class="msg-code-header"><span class="msg-code-lang">${label}</span></div>` +
        `<pre class="msg-code-pre">${body}</pre>` +
      `</div>`
    );
  }

  private escapeHtml(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  private highlightSql(sql: string): string {
    const KW = [
      'SELECT','FROM','WHERE','JOIN','LEFT JOIN','RIGHT JOIN','INNER JOIN','OUTER JOIN',
      'FULL JOIN','CROSS JOIN','ON','GROUP BY','ORDER BY','HAVING','LIMIT','OFFSET',
      'AND','OR','NOT','IN','BETWEEN','LIKE','ILIKE','AS','DISTINCT','CASE','WHEN',
      'THEN','ELSE','END','WITH','UNION','ALL','NULL','IS','IS NOT','FILTER',
      'PARTITION BY','OVER','INSERT','UPDATE','DELETE','SET','VALUES','EXISTS',
    ];
    const FN = [
      'AVG','COUNT','SUM','MAX','MIN','ROUND','EXTRACT','TO_CHAR','DATE_TRUNC',
      'NULLIF','COALESCE','CAST','LOWER','UPPER','TRIM','LENGTH','CONCAT','NOW',
      'CURRENT_DATE','CURRENT_TIMESTAMP','INTERVAL','ARRAY_AGG','STRING_AGG',
      'ROW_NUMBER','RANK','DENSE_RANK','LEAD','LAG',
    ];
    // Process in safe order: escape HTML → strings → comments → functions → keywords → aliases
    let h = this.escapeHtml(sql);
    // String literals (single-quoted)
    h = h.replace(/'([^']*)'/g, `<span class="ss">'$1'</span>`);
    // Line comments
    h = h.replace(/--.*/g, (m) => `<span class="sc">${m}</span>`);
    // Functions (before keywords so COUNT etc. aren't double-spanned)
    FN.forEach(f => {
      h = h.replace(new RegExp(`\\b${f}\\b`, 'g'), `<span class="sf">${f}</span>`);
    });
    // Keywords
    KW.forEach(k => {
      // Word-boundary won't work for multi-word keywords; use case-insensitive literal match
      h = h.replace(new RegExp(`\\b${k}\\b`, 'gi'), (m) => `<span class="sk">${m.toUpperCase()}</span>`);
    });
    // Table aliases (short identifiers before a dot)
    h = h.replace(/\b([a-z_]{1,8})\./g, (_, a) => `<span class="sa">${a}</span>.`);
    return h;
  }
}
