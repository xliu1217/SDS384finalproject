import csv

lime_list = [
    "Why did the AI identify this bird as this class",
    "Can you point out the features in the image that led to this classification?",
    "What parts of the bird were most influential in the AI's decision?",
    "What characteristics does the AI consider when classifying bird species?",
    "Can you show me which pixels affected the AI's prediction the most?",
    "Why was this particular feature of the bird significant for the AI?",
    "Why did the AI choose this class for the image?",
    "What local features contribute to the AI's decision?",
    "Which pixels influenced the AI's classification the most?",
    "Can you highlight the parts of the image that led to this result?",
    "What are the top factors that the model considered for this image?",
    "What specific aspect of this bird's appearance influenced the AI's classification?",
    "Can you show the key regions in the image that determined the bird's species?",
    "Which part of the bird's plumage was most significant for the AI's analysis?",
    "What visual features make this bird distinct from others in the AI's view?",
    "Can you identify the pixels that led the AI to classify this bird as this class?",
    "What contrast or texture features in the bird's image were pivotal for its classification?",
    "Could you highlight the areas of the image that were crucial for the AI's decision?",
    "Which features of the bird's image align with typical characteristics of its species, according to the AI?",
    "How does the AI weigh different visual elements of the bird in its classification process?",
    "What are the subtle differences in the bird's image that the AI considers important for its classification?",
    "In what ways did the lighting or angle of the image influence the AI's interpretation?"
    "What specific pattern in the bird's feathers was a key factor in the AI's decision?",
    "How did the shape of the bird's beak influence its classification by the AI?",
    "Can you point out the features that distinguish this bird from other similar species in the AI's analysis?",
    "How does the AI interpret different textures in the bird's image for classification?",
    "Can you illustrate the effect of shadows or highlights on the AI's analysis of the bird?",
    "What elements in the bird's surrounding environment were considered significant by the AI?",
    "How did the AI assess the bird's wing pattern in determining its species?",
    "What characteristics of the bird's eye and head area were pivotal for the AI's classification?",
    "Can you show how the AI differentiates this bird from others based on color distribution in the image?",
    "How does the AI evaluate the bird's tail features for classification purposes?",
    "Can you demonstrate how the AI processes the bird's leg and feet characteristics for identification?",
    "What role did the bird's movement or stillness in the image play in the AI's decision-making?"
]

shap_list = [
    "What is the overall contribution of each feature to the AI's decision?",
    "Can you break down the prediction into a sum of effects from individual features?",
    "How do each of the image features impact the model's output?",
    "Which features increase the likelihood of this classification?",
    "What are the positive and negative influences in the image on the AI's decision?",
    "How would changing certain features of the bird affect the AI's prediction?",
    "How do different color patterns on the bird contribute to its classification by the AI?",
    "What is the contribution of the bird's wing shape to the overall prediction?",
    "Which features in the image positively and negatively influenced the AI's bird classification?",
    "Can you show the negative and positive aspects of the image that led to the AI's decision?",
    "What elements in the bird's image contribute positively and negatively to the AI's classification?",
    "How do certain parts of the bird's image decrease the likelihood of this particular classification?",
    "What are the contrasting features in the image that led to the AI' predcition?",
    "Which background elements in the image had a positive and negative impact on the AI's prediction?",
    "How do certain colors in the bird's image positively or negatively affect the AI's classification?",
    "What details in the bird's feather pattern are positively and negatively significant in the AI's analysis?",
    "Which aspects of the bird's environment are helping or hindering the AI's decision-making process?",
    "What are the specific attributes of the bird that are both favorably and unfavorably influencing the AI's classification?",
    "How do the contrasting patterns in the bird's plumage affect the AI's decision in both positive and negative ways?",
    "Which features in the image decrease the AI's confidence in its prediction?",
    "What aspects of the bird's appearance reduce the likelihood of this classification?",
    "What details in the image are most detrimental to the AI's classification accuracy?",
    "How do certain patterns or colors in the bird's image negatively impact the AI's prediction?",
    "Which parts of the bird's image seem to confuse or challenge the AI model?",
    "Can you highlight the areas in the image that detract from the AI's classification certainty?",
    "What elements of the bird's environment are causing uncertainty in the AI's decision?",
    "How do shadows or lighting conditions in the image impact the AI's prediction?",
    "What characteristics of the bird are likely causing misclassification by the AI?",
    "Which textures or patterns in the image are leading to lower confidence in the AI's prediction?",
    "Are there specific features in the bird's posture that decrease the AI's classification accuracy?",
    "What are the factors in the bird's surroundings that negatively influence the AI's decision?",
    "Which attributes of the bird are most likely to be misinterpreted by the AI model?"
]

counter_list = [
    "What minimal change would lead the AI to classify this bird differently?",
    "If one aspect of this bird were different, what would the AI predict?",
    "Can you show a slightly altered image where the bird is classified as another species?",
    "Which small modifications in the image would result in a different AI prediction?",
    "How would the classification change if the bird's color was slightly different?",
    "What alterations to the bird's features would lead to a different AI outcome?",
    "What if this bird had a different pattern on its feathers? How would the AI classify it?",
    "Can you generate an image where a tiny change leads to a new AI classification?",
    "What feature changes in the image could mislead the AI into a different prediction?",
     "What is the closest bird species to the predicted one, and how are they different?",
    "Can you show me a bird that looks similar but is classified differently?",
    "What similar bird species could the AI have predicted instead?",
    "Are there any closely related species that the AI might confuse with this one?",
    "Which bird species has the most similar features but is a different type?",
    "Can you provide an example of a bird that is visually similar but belongs to another class?",
    "What other bird species shares similar characteristics with the predicted species?",
    "Is there a bird species that closely resembles this one in appearance but is categorized differently?",
    "Can the AI show a similar-looking bird that falls under a different classification?",
    "What is an almost identical bird species to this one, but with a different name?",
    "Which bird would be the next closest match in the AI's prediction if it wasn't this species?",
    "Can you display a bird that is nearly identical but still distinct from the predicted species?",
    "What would be an alternative classification for a bird with very similar features?",
    "Are there visually similar species that might be easily mistaken for this one?",
    "Which bird species looks nearly the same but is identified differently by the AI?",
     "Show me a bird that's almost the same but classified under a different species.",
    "What minor changes in features would lead to a different bird classification?",
    "If the color or pattern were slightly different, what species might it be?",
    "Display a bird that's visually close but has a key difference in species.",
    "What's the nearest species the AI could confuse this bird with?",
    "Could you illustrate a scenario where a small tweak changes the bird's classification?",
    "Show a similar species where just one or two features differ.",
    "If one key characteristic was altered, which species would this bird resemble?",
    "Can you provide an instance of a bird that shares most features but is a different type?",
    "Which bird is a close match in appearance but not the same species?",
    "Illustrate a bird species that is a near miss in classification compared to this one.",
    "Can you demonstrate an almost identical species with a different name?",
    "Show a bird species that only differs slightly in appearance from this one.",
    "What would be the next best guess for this bird if not for one feature difference?",
    "Display a bird species that's a close contender but has a different classification."
]

same_list = [
    "Can I see more examples of this class from the gallery?",
    "Are there other images that the AI classified similarly?",
    "Show me more images that resulted in the same AI prediction.",
    "What do other instances of this predicted class look like?",
    "Could you display additional images that fall under this category?",
    "Could you provide a variety of images of this bird species?",
    "Are there more photos available showing different features of this bird type?",
    "Show me variations of this bird species in your gallery.",
    "Can you display a range of images from the same bird classification?",
    "I’d like to explore more examples of birds that are classified like this one.",
    "Do you have a collection of images of this specific bird type?",
    "Can we view more pictures that fall within the same classification as this bird?"
     "Can you provide more images of this bird in different positions?",
    "Show me this bird species with various backgrounds.",
    "I'd like to see this bird from different angles. Can you display that?",
    "Please present more photos of this bird in different poses.",
    "Can you show this bird species in a variety of natural settings?",
    "I'm interested in seeing this bird with different plumage. Do you have images?",
    "Are there pictures of this bird species in flight?",
    "Could you display this bird in various stages of its life cycle?",
    "Show me this bird species in different seasons or lighting.",
    "Can you provide images of this bird engaging in different activities?",
    "I'd like to see more examples of this bird's behavior. Can you show that?",
    "Please present this bird species interacting with its environment.",
    "Can you display more photos showing the size comparison of this bird with other objects?",
    "Show this bird species in diverse geographical locations, if possible.",
    "I'd like to see more images that highlight unique features of this bird."
]

data = []

for question in lime_list:
    data.append({"question": question, "label": "LIME"})

for question in shap_list:
    data.append({"question": question, "label": "SHAP"})

for question in counter_list:
    data.append({"question": question, "label": "Counterfactual"})

for question in same_list:
    data.append({"question": question, "label": "SameClass"})

xai_dataset = "/home/zhutou/project/xai_visualization/previous/xai_dataset.csv"

# Write data to CSV
with open(xai_dataset, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=["question", "label"])
    writer.writeheader()
    writer.writerows(data)

print(f"Data successfully written to {xai_dataset}")
