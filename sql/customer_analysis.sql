SELECT
    segment_name,
    COUNT(*) AS customers,
    AVG(recency_days) AS avg_recency_days,
    AVG(frequency) AS avg_frequency,
    AVG(monetary) AS avg_monetary,
    AVG(avg_order_value) AS avg_order_value
FROM customer_segments
GROUP BY segment_name
ORDER BY avg_monetary DESC;
