WITH scored AS (
  SELECT split, target, flagged FROM evaluation WHERE score IS NOT NULL
)
SELECT split, COUNT(*) AS n, SUM(target) AS positives,
       SUM(CASE WHEN target=1 AND flagged=1 THEN 1 ELSE 0 END) AS tp,
       SUM(CASE WHEN target=0 AND flagged=1 THEN 1 ELSE 0 END) AS fp,
       SUM(flagged) AS flagged
FROM scored GROUP BY split ORDER BY split;
