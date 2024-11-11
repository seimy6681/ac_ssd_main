import torch
import torch.nn as nn

import hangul_jamo
# import word_difficulties

# detecting final consonants in target words 8/5

def has_final_consonant(syllable):
    # Unicode for Hangul Syllables starts from 0xAC00
    base_code = 0xAC00
    syllable_code = ord(syllable) - base_code

    if 0 <= syllable_code <= (0xD7A3 - base_code):
        final_consonant_index = syllable_code % 28
        return final_consonant_index != 0
    else:
        raise ValueError("The input character is not a valid Hangul syllable.")

def detect_final_consonants(word):
    final_consonants = []
    for char in word:
        try:
            if has_final_consonant(char):
                final_consonants.append(char)
        except ValueError:
            # Non-Hangul characters will be ignored
            pass
    return final_consonants

# Test with the example word "딸기"
# word = "딸낍"
# final_consonants = detect_final_consonants(word)
# print("Final consonants:", final_consonants)

def calculate_difficulty(word):
 
    score = 0
    
    # count NUMBER OF SYLLABLES
    syllables = len(word)
    if syllables == 1:
        score = 1 # if one syllable(컵, 책), subtract 1 difficulty point
    elif syllables > 1:
        score += syllables
        # print('syllabic: current score: ', score)
    
    # count NUMBER OF FINAL CONSONANTS (받침)
    final_cons = len(detect_final_consonants(word))
    score += final_cons*2
    
    # decompose consonants and vowels
    word = hangul_jamo.decompose(word)
    
    # 쌍자음 Double consonants (ㄲ, ㄸ, ㅃ, ㅆ, ㅉ)
    double_consonants = ['ㄲ', 'ㄸ' , 'ㅃ', 'ㅆ', 'ㅉ']
    score += sum(1 for char in word if char in double_consonants)
    # print('double cons: current score: ', score)
    
    # 격음 Aspirated consonants (ㅋ, ㅌ, ㅊ, ㅍ)
    aspirated_consonants = ['ㅋ', 'ㅌ', 'ㅊ', 'ㅍ']
    score += sum(1 for char in word if char in aspirated_consonants)
    # print('aspirated: current score: ', score)
    
    double_vowels = ['ㅕ', 'ㅑ', 'ㅖ', 'ㅛ', 'ㅠ', 'ㅢ', 'ㅘ', 'ㅝ', 'ㅙ', 'ㅞ']
    score += sum(1 for char in word if char in double_vowels)
    
    # ㄹ consonant
    if 'ㄹ' in word:
        score += 1
        # print('ㄹ current score: ', score)
    
    stressed = ['색종이', '옥수수']
    less_common = ['크레파스', '도깨비', '미끄럼틀', '엘리베이터', '파인애플']
    if word in less_common:
        score += 1
           
    if score < 0:
        score = 0    
    return score

# print(calculate_difficulty('딸기'))
words = ["포도", "딸기", "사탕", "햄버거", "옥수수", "컵", "빨대", "책", "색종이", "머리", "양말", "단추", "모자", "장갑", "빗", "우산", "침대", "화장실", "나무", "꽃", "바퀴", "그네", "시소", "눈사람", "토끼", "이빨", "거북이", "뱀", "호랑이", "고래", "찢어요", "싸워요", "아파요", "병원", "안경", "없어요", "올라가요", "빵", "쨈", "쌀", "풀", "학", "총", "달", "집", "문", "눈", "턱", "똥", "김", "손", "링", "입", "코", "별", "오이", "엄마", "악어", "새우", "뽀뽀", "치즈", "매미", "두개", "가방", "도둑", "라면", "밥통", "풍선", "도깨비", "바나나", "피아노", "떡볶이", "목욕탕", "선생님", "할아버지", "크레파스", "파인애플", "미끄럼틀", "엘리베이터", "아이스크림"]


word_difficulties = {word: calculate_difficulty(word) for word in words}

# Sort words by difficulty
sorted_words = sorted(word_difficulties.items(), key=lambda x: x[1])

# Print results
for word, difficulty in sorted_words:
    print(f"{word}: {difficulty}")

sorted_words_dict_content = f"word_difficulties = {dict(sorted_words)}"

# Save the dictionary to a .py file
with open('word_difficulty_list.py', 'w', encoding='utf-8') as f:
    f.write(sorted_words_dict_content)

print("Sorted words have been saved to 'sorted_word_difficulties.py'")

def get_difficulty_factor(word, is_disordered, word_difficulties):
    difficulty_score = calculate_difficulty(word)

    # Determine if the word is easy or difficult
    threshold = 2  # You can adjust this threshold based on your criteria
    is_easy_to_pronounce = difficulty_score < threshold

    # Determine the difficulty factor based on the word's difficulty and speaker type
    if is_easy_to_pronounce:
        if is_disordered:
            difficulty_factor = 1.5  # Higher difficulty for disordered speaker for easy words
        else:
            difficulty_factor = 0.5  # Lower difficulty for normal speaker for easy words
    else:
        if is_disordered:
            difficulty_factor = 0.5  # Lower difficulty for disordered speaker for difficult words
        else:
            difficulty_factor = 1.5  # Higher difficulty for normal speaker for difficult words

    return difficulty_factor

# print('hi')
# print(get_difficulty_factor('딸기', True, word_difficulties))
# print(get_difficulty_factor('딸기', False, word_difficulties))
# # print(calculate_difficulty('딸기'))
# for word in words:
#     print(f'for normal, {word}: {get_difficulty_factor(word,0, word_difficulties)}')

#for NORMAL SPEAKER
difficulty_factors_0 = {word: get_difficulty_factor(word,0,word_difficulties) for word in words}
difficulty_factors_0 = sorted(difficulty_factors_0.items(), key=lambda x: x[1])

difficulty_factors_dict_content_0 = f"difficulty_factors_1 = {dict(difficulty_factors_0)}"

# Save the dictionary to a .py file
with open('difficulty_factors_0.py', 'w', encoding='utf-8') as f:
    f.write(difficulty_factors_dict_content_0)
    
#for SSD SPEAKER----------------------------------------------
difficulty_factors_1 = {word: get_difficulty_factor(word,1,word_difficulties) for word in words}
difficulty_factors_1 = sorted(difficulty_factors_1.items(), key=lambda x: x[1])

difficulty_factors_dict_content_1 = f"difficulty_factors_1 = {dict(difficulty_factors_1)}"

# Save the dictionary to a .py file
with open('difficulty_factors_1.py', 'w', encoding='utf-8') as f:
    f.write(difficulty_factors_dict_content_1)