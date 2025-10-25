# EcoJusticeAI
We, as a team of three members, have observed the growing need for an **AI-based system** that ensures **strict regulation on litter detection** to promote **environmental cleanliness** within university campuses and public areas.  

Our project, **EcoJusticeAI**, uses **Computer Vision** and **AI technologies** to automatically detect littering incidents, identify offenders, and help enforce cleanliness policies.

## 🧩 Approach and Challenges

Initially, we used **Roboflow** to train a YOLO-based custom model for detecting multiple classes such as **person**, **litter**, and **face**.  
However, during testing, we noticed that **litter objects were often detected inside the bounding box of the person**, which made it difficult to accurately determine if the person had dropped the litter.  

To overcome this, we adopted a **hybrid approach**:

- 🧍‍♂️ **MediaPipe** is used for **pose detection**, focusing mainly on **hand tracking** to observe user actions.  
- 🗑️ **Roboflow** (YOLO model) is still used for **litter detection**, as it performs well in identifying small objects on the ground.

By combining **pose-based movement analysis** with **object detection**, the system more reliably identifies **real littering incidents** while reducing false positives.

