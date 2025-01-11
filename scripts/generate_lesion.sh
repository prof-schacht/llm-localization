for network in moral random none;
do
    python -m generate_lesion \
        --model-name microsoft/phi-4 \
        --prompt "User: Which statement best represents the moral dimension of 'care'?\nA) It’s acceptable to ignore a stranger’s suffering if it doesn’t affect you.\nB) One should always come to the aid of a stranger in distress. Answer:" \
        --percentage 1 \
        --network $network
done 