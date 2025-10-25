# EcoJusticeAI
We ,as a team of 3 members ,have observed that we need an AI agent which will ensure strict regulation on litter detection which will promote environment cleanliness.
## 🧩 Approach and Challenges

Initially, we used **Roboflow** to train a custom YOLO model for detecting multiple objects such as **person**, **litter**, and **face** in the same frame.  
However, we encountered a major issue — sometimes the **litter object was detected inside the bounding box of the person**, which made it difficult to accurately determine whether the person had dropped the object or not.  

To overcome this challenge, we shifted our approach to using the **MediaPipe API**, which allowed us to detect **human poses**, especially **hand movements**.  
This made it easier to analyze the **action of littering** based on the position of the person’s hand and the movement of the object, improving accuracy and reducing false detections.
