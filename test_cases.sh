#!/bin/bash

texts=(
"Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications."
"ok so i finally tried that new ramen place downtown and honestly? underwhelming. the broth was fine but they put WAY too much sodium in it."
"The relationship between monetary policy and asset price inflation has been extensively studied in the literature."
"I have been thinking a lot about remote work lately. There are genuine tradeoffs, flexibility and no commute on one side, isolation and blurred work-life boundaries on the other."
)

for text in "${texts[@]}"
do
  echo "-----------------------------------"
  curl -s -X POST http://127.0.0.1:5000/submit \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$text\",\"creator_id\":\"test-user-1\"}" \
    | python -m json.tool
done