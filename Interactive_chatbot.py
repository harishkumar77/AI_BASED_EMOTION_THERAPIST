import random
import re
from gtts import gTTS
from playsound import playsound
import pandas as pd
from num2words import num2words
import cv2
import time

reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}


class Chat:
    def _init_(self, pairs, reflections={}):
        """
        Initialize the chatbot.  Pairs is a list of patterns and responses.  Each
        pattern is a regular expression matching the user's statement or question,
        e.g. r'I like (.*)'.  For each such pattern a list of possible responses
        is given, e.g. ['Why do you like %1', 'Did you ever dislike %1'].  Material
        which is matched by parenthesized sections of the patterns (e.g. .*) is mapped to
        the numbered positions in the responses, e.g. %1.

        :type pairs: list of tuple
        :param pairs: The patterns and responses
        :type reflections: dict
        :param reflections: A mapping between first and second person expressions
        :rtype: None
        """

        self._pairs = [(re.compile(x, re.IGNORECASE), y) for (x, y) in pairs]
        self._reflections = reflections
        self._regex = self._compile_reflections()

    def _compile_reflections(self):
        sorted_refl = sorted(self._reflections, key=len, reverse=True)
        return re.compile(
            r"\b({})\b".format("|".join(map(re.escape, sorted_refl))), re.IGNORECASE
        )

    def _substitute(self, str):
        """
        Substitute words in the string, according to the specified reflections,
        e.g. "I'm" -> "you are"

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        return self._regex.sub(
            lambda mo: self._reflections[mo.string[mo.start(): mo.end()]], str.lower()
        )

    def _wildcards(self, response, match):
        pos = response.find("%")
        while pos >= 0:
            num = int(response[pos + 1: pos + 2])
            response = (
                    response[:pos]
                    + self._substitute(match.group(num))
                    + response[pos + 2:]
            )
            pos = response.find("%")
        return response

    def respond(self, str):
        """
        Generate a response to the user input.

        :type str: str
        :param str: The string to be mapped
        :rtype: str
        """

        # check each pattern
        for (pattern, response) in self._pairs:
            match = pattern.match(str)

            # did the pattern match?
            if match:
                resp = random.choice(response)  # pick a random response
                resp = self._wildcards(resp, match)  # process wildcards

                # fix munged punctuation at the end
                if resp[-2:] == "?.":
                    resp = resp[:-2] + "."
                if resp[-2:] == "??":
                    resp = resp[:-2] + "?"
                return resp

    def converse(self, quit="quit"):
        user_input = ""
        while user_input != quit:
            user_input = quit
            try:
                user_input = input(">")
            except EOFError:
                print(user_input)
            if user_input:
                while user_input[-1] in "!.":
                    user_input = user_input[:-1]
                print(self.respond(user_input))


import nltk
# from nltk.chat.util import Chat, reflections
from nltk.chat.util import reflections
from nltk.sentiment import SentimentIntensityAnalyzer
import random

nltk.download('vader_lexicon')
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today?"]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there"]
    ],
    [
        r"what is your name?",
        ["You can call me AI", "My name is AI, what's yours?"]
    ],
    [
        r"how are you?",
        ["I'm doing good", "I am fine, thank you for asking"]
    ],
    [
        r"sorry (.*)",
        ["Its alright", "Its OK, never mind"]
    ],
    [
        r"quit",
        ["Thank you for talking with me, Have a great day!"]
    ],
    [
        r"I am feeling sad",
        ["It is normal to feel that way", "If you need to talk, I am here to listen."]
    ],
    [
        r"I am feeling happy",
        ["You deserve this since you are a good human being", "If you need, you can tell me about your happiness"]
    ],
    [
        r"The people I know betrayed me",
        ["It is sad to know this. Do not worry, I am sure that good people will be around you in future",
         "If you need to talk more about this, you can"]
    ],
    [
        r"The parents scolded me for getting less marks",
        ["I know you are sad now. First speak openly with your parents. Tell them about your hardwork in exams",
         "If you need to talk more about this, you can"]
    ],
    [
        r"I facing a lot of stress.",
        [
            "Stress is part of our life. It will be there till you die. For having a good life, it is better if you do yoga, read books, watch movies tv series etc",
            "If you need to talk more about this, you can"]
    ],
    [
        r"The teacher scolded me even though he or she made mistake",
        ["I know it is wrong. If you are facing more problems, complaint to higher officials"]
    ],
    [
        r"I am tired of living.",
        [
            "Please do not say like this. Every human life is valuable. Just think about your favorite things, you will be normal in some time."]
    ],

    [
        r"Feeling overwhelmed and stressed out",
        [
            "It's okay to feel overwhelmed sometimes. Take a break and do something that makes you happy. You will feel better soon."]
    ],

    [r"I am not good enough",
     ["You are good enough just the way you are. Don't compare yourself to others and focus on your own strengths."]],
    [r"I am feeling lost and directionless", [
        "It's okay to feel lost sometimes. Take some time to reflect on your goals and what you want to achieve. You'll find your way."]],
    [r"I am struggling with my mental health", [
        "It's important to take care of your mental health. Consider talking to a therapist or seeking professional help."]],
    [r"I am feeling lonely and isolated", [
        "It's important to connect with others. Reach out to friends or family, or consider joining a social group or club."]],
    [r"I am feeling stuck in my career", [
        "Take some time to reflect on your career goals and what you want to achieve. Consider seeking out new opportunities or further education."]],
    [r"I am feeling overwhelmed with responsibilities",
     ["It's important to prioritize and delegate tasks. Don't be afraid to ask for help when you need it."]],
    [r"I am feeling anxious and stressed about the future", [
        "Focus on the present moment and take things one step at a time. Consider practicing mindfulness or meditation."]],
    [r"I am feeling unmotivated and uninspired",
     ["Take some time to explore new hobbies or interests. You may find something that sparks your passion."]],
    [r"I am feeling like a failure", [
        "Remember that failure is a natural part of the learning process. Use it as an opportunity to grow and improve."]],

    [r"I am feeling overwhelmed with work", [
        "It's important to prioritize your tasks and take breaks when you need them. Don't be afraid to ask for help or delegate tasks."]],
    [r"I am feeling like I'm not making progress in my life", [
        "Take some time to reflect on your goals and what you want to achieve. Break them down into smaller, achievable steps and work towards them."]],
    [r"I am feeling like I'm not good enough at my job", [
        "Remember that everyone makes mistakes and has room for improvement. Focus on your strengths and seek out opportunities for growth and development."]],
    [r"I am feeling like I'm not connecting with others", [
        "Try to find common ground with others and engage in activities or conversations that interest you both. Don't be afraid to initiate conversations or reach out to others."]],
    [r"I am feeling like I'm not being heard or understood", [
        "Practice active listening and try to communicate your thoughts and feelings clearly. Don't be afraid to ask for clarification or repeat yourself if necessary."]],
    [r"I am feeling like I'm not living up to my potential", [
        "Focus on your strengths and seek out opportunities to use them. Don't be afraid to take risks and try new things."]],
    [r"I am feeling like I'm not getting enough support", [
        "Reach out to friends, family, or colleagues for support. Consider joining a support group or seeking professional help if necessary."]],
    [r"I am feeling like I'm not making a difference", [
        "Remember that even small actions can make a big impact. Focus on the positive changes you can make in your own life and the lives of those around you."]],
    [r"I am feeling like I'm not in control of my life", [
        "Focus on the things you can control and let go of the things you can't. Set goals and work towards them, but be flexible and open to change."]],
    [r"I am feeling like I'm not taking care of myself", [
        "Make self-care a priority and take time to do things that make you happy and relaxed. Practice mindfulness or meditation to reduce stress."]],
    [r"I am feeling like I'm not creative enough", [
        "Try new things and explore different hobbies or interests. Don't be afraid to take risks and think outside the box."]],
    [r"I am feeling like I'm not learning enough", [
        "Seek out opportunities for learning and growth, such as taking classes or attending workshops. Read books or watch videos on topics that interest you."]],
    [r"I am feeling like I'm not appreciated", [
        "Remember that your worth is not determined by others' opinions of you. Focus on your own accomplishments and seek out positive feedback from others."]],
    [r"I am feeling like I'm not making enough money", [
        "Consider seeking out new job opportunities or negotiating a raise. Focus on your skills and experience and be confident in your worth."]],
    [r"I am feeling like I'm not making enough time for my loved ones", [
        "Make time for your loved ones a priority and schedule regular activities or outings with them. Communicate your love and appreciation for them regularly."]],
    [r"I am feeling like I'm not making a positive impact on the world", [
        "Find ways to volunteer or get involved in your community. Focus on making small changes that can have a big impact."]],
    [r"I am feeling like I'm not taking enough risks", [
        "Don't be afraid to take risks and try new things. Embrace failure as a learning opportunity and focus on the potential rewards."]],

]


def chatbot():
    face_cascade = cv2.CascadeClassifier('E:/downloads/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite('E:/downloads/detected_face.jpg', frame)
        cv2.imshow('frame', frame)
        if len(faces) > 0:
            first = "Hey i can see you,i am your AI therapist. How can I help you today?"
            language = 'en'
            obj = gTTS(text=first, lang=language, slow=False)
            obj.save("exam.mp3")
            playsound("exam.mp3")
            print("Hi, I am your AI therapist. How can I help you today?")
            sia = SentimentIntensityAnalyzer()

            # Get user input
            user_input = input("How are you feeling today? ")

            # Analyze the sentiment of the user's input
            sentiment_score = sia.polarity_scores(user_input)
            b = "It sounds like you're feeling positive today,I wish more good things will happen to you."
            # Check the sentiment score and respond accordingly
            if sentiment_score["compound"] >= 0.5:
                obj2 = gTTS(text=b, lang=language, slow=False)
                obj2.save("exam2.mp3")
                playsound("exam2.mp3")
                print("It sounds like you're feeling positive today.")
                print("I wish more good things will happen to you.")
            elif sentiment_score["compound"] > 0 and sentiment_score["compound"] < 0.5:
                c = "It sounds like you're feeling neutral today."
                obj3 = gTTS(text=c, lang=language, slow=False)
                obj3.save("exam3.mp3")
                playsound("exam3.mp3")
                print("It sounds like you're feeling neutral today.")
            else:
                d = "It sounds like you're feeling negative today,So can we play some games or should we chat? or need youtube videos recommendations"
                obj4 = gTTS(text=d, lang=language, slow=False)
                obj4.save("exam4.mp3")
                playsound("exam4.mp3")
                print("It sounds like you're feeling negative today.")
                print("So can we play some games or should we chat? or need youtube videos recommendations")
                a = input()
                if a == 'chat':
                    m = "okay, now we can have some conversation through text"
                    obj12 = gTTS(text=m, lang=language, slow=False)
                    obj12.save("E:/downloads/exam12.mp3")
                    playsound("E:/downloads/exam12.mp3")
                    chat = Chat(pairs, reflections)
                    chat.converse()
                elif a == 'youtube' or a == 'videos':
                    n = "okay, here is some of my youtube recommendations which can possibly change your mood"
                    obj13 = gTTS(text=n, lang=language, slow=False)
                    obj13.save("E:/downloads/exam13.mp3")
                    playsound("E:/downloads/exam13.mp3")
                    df = pd.read_excel('/content/E2.xlsx')
                    # extract the first column
                    first_column = df.iloc[:, 0]
                    t = df.iloc[:, 1]
                    for row in range(0, 118):
                        sentiment_score = sia.polarity_scores(first_column[row])
                        if sentiment_score["compound"] >= 0.5:
                            print(t[row])
                else:
                    lst1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                    chances_1 = 20
                    no_of_chances_1 = 0
                    your_runs = 0
                    e = "Okay, let's play hand cricket to recover from your stress"
                    obj5 = gTTS(text=e, lang=language, slow=False)
                    obj5.save("E:/downloads/exam5.mp3")
                    playsound("E:/downloads/exam5.mp3")
                    print("-----------------------------------------------\nYour Batting\n")
                    while no_of_chances_1 < chances_1:
                        runs = int(input("Enter Runs for Your Batting Turn: "))
                        comp_bowl = random.choice(lst1)

                        if runs == comp_bowl:
                            print("Your Guess: ", runs, ",Computer Guess: ", comp_bowl)
                            f = "You are Out,haha"
                            obj6 = gTTS(text=f, lang=language, slow=False)
                            obj6.save("exam6.mp3")
                            playsound("exam6.mp3")
                            print("You are Out. Your Total Runs= ", your_runs, "\n")
                            break
                        elif runs > 10:
                            g = "sorry the game will support only upto 10"
                            obj7 = gTTS(text=g, lang=language, slow=False)
                            obj7.save("E:/downloads/exam7.mp3")
                            playsound("E:/downloads/exam7.mp3")
                            print("ALERT!! Support No only till 10\n")
                            continue
                        else:
                            your_runs = your_runs + runs
                            run_change = num2words(your_runs)
                            h = "Your runs now are " + run_change
                            obj8 = gTTS(text=h, lang=language, slow=False)
                            obj8.save("E:/downloads/exam8.mp3")
                            playsound("E:/downloads/exam8.mp3")
                            print("Your Guess: ", runs, ",Computer Guess: ", comp_bowl)
                            print("Your runs Now are: ", your_runs, "\n")

                        no_of_chances_1 = no_of_chances_1 + 1

                    lst2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

                    chances_2 = 20
                    no_of_chances_2 = 0
                    comp_runs = 0
                    i = "Now its my time for batting"
                    obj9 = gTTS(text=i, lang=language, slow=False)
                    obj9.save("E:/downloads/exam9.mp3")
                    playsound("E:/downloads/exam9.mp3")
                    print("-----------------------------------------------")
                    print("Computer Batting-\n")
                    while no_of_chances_2 < chances_2:

                        bowl = int(input("Enter Runs for Your Bowling Turn: "))
                        comp_bat = random.choice(lst2)

                        if comp_bat == bowl:
                            print("Computer Guess: ", comp_bat, "Your Guess: ", bowl)
                            print("The Computer is Out. Computer Runs= ", comp_runs, "\n")
                            break
                        else:
                            comp_runs = comp_runs + comp_bat
                            print("Computer Guess: ", comp_bat, "Your Guess: ", bowl)
                            print("Computer Runs: ", comp_runs, "\n")

                            if comp_runs > your_runs:
                                break

                        no_of_chances_2 = no_of_chances_2 + 1

                    print("\n-----------------------------------------------\nRESULTS: ")

                    if comp_runs < your_runs:
                        j = "oh you win,it's nice talking to i hope you recovered from your stress"
                        obj10 = gTTS(text=j, lang=language, slow=False)
                        obj10.save("E:/downloads/exam10.mp3")
                        playsound("E:/downloads/exam10.mp3")
                        print("\nYou won the Game.\n\nYour Total Runs= ", your_runs, "  [Bowls taken(Out of 20): ",
                              no_of_chances_1 + 1, "]", "\nComputer Total Runs= ", comp_runs,
                              "  [Bowls Taken(Out of 20): ",
                              no_of_chances_2 + 1, "]\n")

                    elif comp_runs == your_runs:
                        print("The Game is a Tie")

                    else:
                        k = "yes i own,it's nice talking to you and i hope you recovered from your stress"
                        obj11 = gTTS(text=k, lang=language, slow=False)
                        obj11.save("E:/downloads/exam11.mp3")
                        playsound("E:/downloads/exam11.mp3")
                        print("\nComputer won the Game.\n\nComputer Total Runs= ", comp_runs,
                              "  [Bowls Taken(Out of 20): ",
                              no_of_chances_2 + 1, "]", "\nYour Total Runs= ", your_runs, "  [Bowls taken(Out of 20): ",
                              no_of_chances_1 + 1, "]\n")

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        if len(faces) < 0:
            last = "sorry i couldnt able to see u"
            obj_last = gTTS(text=last, lang=language, slow=False)
            obj_last.save("exam_last.mp3")
            playsound("exam_last.mp3")
if _name_ == "_main_":
    chatbot()

