#!/usr/bin/env python3
"""
Simplified Prompt Manager following ALCE research approach
Much cleaner and more focused than the original verbose approach
"""

from typing import Dict, List, Any
from enum import Enum

class MethodType(Enum):
    POST_RETRIEVAL = "post-retrieval"
    POST_GENERATION = "post-generation"
    POST_GENERATION_LLM_SHORT = "post-generation-llm-short"
    POST_GENERATION_LLM_LONG = "post-generation-llm-long"

class SimplifiedPromptManager:
    """Simplified prompt manager following ALCE research approach"""
    
    def __init__(self):
        self.dataset_instructions = {
            'asqa': "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
            
            'eli5': "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
            
            'hagrid': "Instruction: Write an accurate, concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
            
            'msmarco': "Instruction: Answer with a single word or phrase when possible for the given question using only the provided passages (some of which might be irrelevant). Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
            
            'qampari': "Instruction: Provide a list of accurate answers for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Always cite one and only one document for each answer. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.",
                        
            'natural_questions': "Instruction: Answer with a single short phrase using only the provided passages. Focus on the most relevant information and provide a direct answer. Always cite for any factual claim. Cite only one document in each sentence. If multiple documents support the sentence, only cite the most relevant one."
        }

        # Clean instructions without citation parts (for post-generation)
        self.clean_instructions = {
            'asqa': "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone.",
            
            'eli5': "Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone",
            
            'hagrid': "Instruction: Write an accurate, concise answer for the given question. Use an unbiased and journalistic tone.",
            
            'msmarco': "Instruction: Answer with a single word or phrase when possible for the given question.",
            
            'qampari': "Instruction: Provide a list of accurate answers for the given question. Separate answers by commas. For questions that have more than 5 answers, write at least 5 answers.",
                        
            'natural_questions': "Instruction: Answer with a single short phrase."
        }
        
        # Few-shot examples for all datasets (configurable number)
        self.few_shot_examples = {
            'asqa': [
                {
                    "question": "Which is the most rainy place on earth?",
                    "answer": "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3]. However, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3], although nearby town Sohra, India, also known as Cherrapunji, holds the record for most rain in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 [1].",
                    "docs": [
                        {
                            "title": "Cherrapunji",
                            "text": "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in"
                        },
                        {
                            "title": "Cherrapunji",
                            "text": "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall"
                        },
                        {
                            "title": "Mawsynram",
                            "text": "Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′"
                        },
                        {
                            "title": "Earth rainfall climatology",
                            "text": "Pacific Northwest, and the Sierra Nevada range are the wetter portions of the nation, with average rainfall exceeding per year. The drier areas are the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern Utah, central Wyoming, eastern Oregon and Washington and the northeast of the Olympic Peninsula. The Big Bog on the island of Maui receives, on average, every year, making it the wettest location in the US, and all of Oceania. The annual average rainfall maxima across the continent lie across the northwest from northwest Brazil into northern Peru, Colombia, and Ecuador, then along the Atlantic coast of"
                        },
                        {
                            "title": "Going to Extremes",
                            "text": "in the world. Oymyakon in Siberia, where the average winter temperature is −47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive years without rain. Fog is the only local source of water. Mawsynram in India, where average annual rainfall is 14 meters, falling within a four-month period in the monsoon season. The rainfall is approximately equal to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the 'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over the year. In his second series, Middleton visited places without permanent towns, locations where \"survival\""
                        }
                    ]
                },
                {
                    "question": "When did the us break away from england?",
                    "answer": "The United States took the first step towards gaining independence from Great Britain when it declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, 1776, the date when the Declaration of Independence was officially adopted by Congress) [2]. The Treaty of Paris was later signed on September 3, 1783, formally separating the United States from the British Empire [3].",
                    "docs": [
                        {
                            "title": "United States withdrawal from Saudi Arabia",
                            "text": "United States withdrawal from Saudi Arabia Beginning during Operation Desert Shield in August 1990, while preparing for the Gulf War, the United States sent a large troop contingent to Saudi Arabia. After the war, remnant troops, primarily U.S. Air Force personnel, augmented by a smaller number of coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The United Kingdom and France also maintained a small contingent of Royal Air Force and French Air Force"
                        },
                        {
                            "title": "Decolonization of the Americas",
                            "text": "and France has fully \"integrated\" most of its former colonies as fully constituent \"departments\" of France. The United States of America declared independence from Great Britain on July 2, 1776 (although the event is now commemorated on July 4, the date when the Declaration of Independence was officially adopted by Congress), in so doing becoming the first independent, foreign-recognized nation in the Americas and the first European colonial entity to break from its mother country. Britain formally acknowledged American independence in 1783 after its defeat in the American Revolutionary War. Although initially occupying only the land east of the Mississippi"
                        },
                        {
                            "title": "American Revolution",
                            "text": "second British army at Yorktown in the fall of 1781, effectively ending the war. The Treaty of Paris was signed September 3, 1783, formally ending the conflict and confirming the new nation's complete separation from the British Empire. The United States took possession of nearly all the territory east of the Mississippi River and south of the Great Lakes, with the British retaining control of Canada and Spain taking Florida. Among the significant results of the revolution was the creation of the United States Constitution, establishing a relatively strong federal national government that included an executive, a national judiciary, and"
                        },
                        {
                            "title": "Decolonization",
                            "text": "accelerate decolonialization and bring an end to the colonial empires of its Western allies, most importantly during the 1956 Suez Crisis, but American military bases were established around the world and direct and indirect interventions continued in Korea, Indochina, Latin America (\"inter alia\", the 1965 occupation of the Dominican Republic), Africa, and the Middle East to oppose Communist invasions and insurgencies. Since the dissolution of the Soviet Union, the United States has been far less active in the Americas, but invaded Afghanistan and Iraq following the September 11 attacks in 2001, establishing army and air bases in Central Asia. Before"
                        },
                        {
                            "title": "Decolonization",
                            "text": "the responsibility of the United Kingdom (with a copy of the new constitution annexed), and finally, if approved, issuance of an Order of Council fixing the exact date of independence. After World War I, several former German and Ottoman territories in the Middle East, Africa, and the Pacific were governed by the UK as League of Nations mandates. Some were administered directly by the UK, and others by British dominions – Nauru and the Territory of New Guinea by Australia, South West Africa by the Union of South Africa, and Western Samoa by New Zealand. Egypt became independent in 1922,"
                        }
                    ]
                },
                {
                    "question": "Who set the record for longest field goal?",
                    "answer": "The record for the longest field goal in an NFL game was set by Matt Prater at 64 yards [1], but the record for the longest field goal at any level was 69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene Christian University football game against East Texas State University [2].",
                    "docs": [
                        {
                            "title": "Field goal",
                            "text": "toward its own end. The longest field goal kick in NFL history is 64 yards, a record set by Matt Prater on December 8, 2013. The previous record was 63, originally set by Tom Dempsey (1970) and then matched by Jason Elam (1998), Sebastian Janikowski (2011), David Akers (2012), and Graham Gano (2018). High school, college and most professional football leagues offer only a three-point field goal; however, some professional leagues have encouraged more rare kicks through \"four-point field goals\". NFL Europe encouraged long field goals of 50 yards or more by making those worth four points instead of three"
                        },
                        {
                            "title": "Field goal range",
                            "text": "35 and 40 yard lines (closer in a crosswind) often will go for the more risky fourth down conversion rather than risk either the touchback or the missed field goal. The longest field goal in recorded football history was 69 yards, set by collegiate kicker Ove Johansson, who was born in Sweden, in a 1976 Abilene Christian University football game against East Texas State University (now Texas A&M Commerce) at Shotwell Stadium in Abilene. The longest successful field goal in the NFL was 64 yards and was completed by Matt Prater in 2013. The NCAA record is 67 yards held"
                        },
                        {
                            "title": "Field goal",
                            "text": "both end zones) is only 66 yards. Scaccia, while playing indoor football, attempted a 64-yard kick that was inches short of success, hitting the crossbar. Longer field goals have been attempted at times; the longest attempt in the NFL, which was well short and was kicked into the wind, was 76 yards, attempted by Sebastian Janikowski of the Oakland Raiders, in a September 28, 2008 game against the San Diego Chargers. NFL Europe rewarded kickers that successfully kicked a field goal of longer than 50 yards with a bonus point, making such field goals worth 4 points instead of 3;"
                        },
                        {
                            "title": "Field goal",
                            "text": "this accomplishment is not the official record. All of the above kicks were successful with the use of a kicking tee, which was banned by the NCAA after the 1988 season. The longest known drop-kicked field goal in college football was a 62-yard kick from Pat O'Dea, an Australian kicker who played on the Wisconsin Badgers football team. O'Dea's kick took place in a blizzard against Northwestern on November 15, 1898. The longest field goal in U Sports football history is 59 yards, by Niko Difonte of Calgary Dinos, playing against the UBC Thunderbirds on November 11, 2017. The field"
                        },
                        {
                            "title": "Field goal range",
                            "text": "NFL and have been banned from NCAA since 1989) is 68 yards held by Fabrizio Scaccia, and the high school record 68 yards held by Dirk Borgognone; high school has wider goal posts and treats a field goal attempt that lands short in the field of play the same as a punt, making longer attempts much less risky. The indoor football record, with narrower and higher goal posts, is 63 yards (set by Aaron Mills), which is practically as long of a field goal as is possible in that variant of the sport, since the field in indoor football (including"
                        }
                    ]
                },
                {
                    "question": "Who played galen in planet of the apes?",
                    "answer": "In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. And in the tv series Planet of the Apes, Galen was played by Roddy McDowall [1].",
                    "docs": [
                        {
                            "title": "Planet of the Apes",
                            "text": "installment. Jacobs died on June 27, 1973, bringing an end to the APJAC Productions era of the \"Planet of the Apes\" franchise. Former Fox executive Stan Hough took over as producer for the television project, titled \"Planet of the Apes\". CBS picked up the series for its 1974 autumn lineup. Ron Harper and James Naughton played Alan Virdon and Peter Burke, two 20th-century American astronauts who pass through a time warp to a future where apes subjugate humans (unlike the original film, the humans can speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who joins the astronauts."
                        },
                        {
                            "title": "Planet of the Apes (1968 film)",
                            "text": "chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright King). While unable to speak as his throat wound is healing, called \"Bright Eyes\" by Zira and placed with one of the captive primitive humans he later names \"Nova\", Taylor observes the enhanced society of talking apes and in a strict caste system: the gorillas being the military police, hunters and workers; the orangutans overseeing the affairs of government, science, and religion; and intellectual chimpanzees being mostly scientists. While their society is a theocracy similar to the beginnings of the human Industrial Era, the apes consider the primitive humans as"
                        },
                        {
                            "title": "Planet of the Apes (1968 film)",
                            "text": "Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely based on the 1963 French novel \"La Plan\u00e8te des Singes\" by Pierre Boulle. Jerry Goldsmith composed the groundbreaking avant-garde score. It was the first in a series of five films made between 1968 and 1973, all produced by Arthur P. Jacobs and released by 20th Century Fox. The film tells the"
                        },
                        {
                            "title": "Planet of the Apes",
                            "text": "Rupert Wyatt. To portray ape characters realistically, the production avoided practical effects in favor of performance capture acting, partnering with New Zealand visual effects company Weta Digital. Wyatt cast James Franco as Will Rodman, while veteran performance capture actor Andy Serkis signed on to star as Caesar. \"Rise\" debuted on August 5, 2011. Critics reviewed it positively, especially praising the visual effects and Serkis's performance. It was a major box office hit, taking in $482 million globally, more than five times its $93 million budget. Weta's special effects earned the film two Visual Effects Society Awards and an Oscar nomination"
                        },
                        {
                            "title": "Planet of the Apes",
                            "text": "film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels through a wormhole to a distant planet where talking apes enslave humans. He leads a human revolt and upends ape civilization by discovering that the apes evolved from the normal earth primates who had accompanied his mission, and arrived years before. Helena Bonham Carter played chimpanzee Ari, while Tim Roth played the human-hating chimpanzee General Thade. The film received mixed reviews; most critics believed it failed to compare to the original. Much of the negative commentary focused on the confusing plot and twist ending, though many reviewers praised the"
                        }
                    ]
                }
            ],
            
            'eli5': [
                {
                    "question": "Why did New York City try to ban food donations to the poor?",
                    "answer": "New York City, under Mayor Michael Bloomberg's administration, banned citizens from donating food directly to homeless shelters because the city could not assess the salt, fat, and fiber content [1][2][3]. Bloomberg's administration was heavily criticized for losing their common sense by becoming too focused on what people eat [2].",
                    "docs": [
                        {
                            "title": "The Future Of America", 
                            "text": "believe that they are \u201chelping\u201d the homeless by passing such laws. In New York City, Mayor Bloomberg has banned citizens from donating food directly to homeless shelters and he is actually convinced that it was the right thing to do for the homeless\u2026 Mayor Michael Bloomberg\u2019s food police have struck again! Outlawed are food donations to homeless shelters because the city can\u2019t assess their salt, fat and fiber content, reports CBS 2\u2019s Marcia Kramer. Glenn Richter arrived at a West Side synagogue on Monday to collect surplus bagels \u2014 fresh nutritious bagels \u2014 to donate to the poor."
                        },
                        {
                            "title": "mayor bloomberg", 
                            "text": "Amuck: Bloomberg Bans Food Donations in New York City Food Might Be Salty or Too High in Calories, City Explains Washington, D.C. \u2013 New York Mayor Michael Bloomberg\u2019s administration is now banning all food being offered to the city\u2019s homeless shelters. New York City\u2019s bureaucrats have become so singularly focused on what people eat, says the National Center for Public Policy Research, that they\u2019ve lost their common sense. \u201cSo much for serving the homeless: The Bloomberg administration is now taking the term \u2018food police\u2019 to new depths, blocking food donations to all government-run facilities that serve the"
                        },
                        {
                            "title": "New York City bans food donations - WND", 
                            "text": "New York City bans food donations - WND Front Page Health U.S. New York City bans food donations Inability to control 'nutritional content' cited as reason New York City homeless shelters have Mayor Michael Bloomberg to thank for a halt in food donations, for which hungry families are waiting, according to one public policy advocate. \"The Bloomberg administration is now taking the term 'food police' to new depths, blocking food donations to all government-run facilities that serve the city's homeless,\" says Jeff Stier, a National Center for Public Policy Research senior fellow. Currently, no food can be given to government-run, New York City facilities, despite hungry crowds perfectly"
                        },
                        {
                            "title": "New York City bans food donations - WND", 
                            "text": "New York City bans food donations - WND Services didn't return WND calls. Stier told WND that he specifically was told by Diamond that the policy was tied to the nutritional guidelines set by the mayor. \"They can say that this ban on donations is a long-standing policy, but they can\u2019t document it,\" Stier told WND. \"I've also been told that there are numerous food shelves that have been accepting food donations, not just one.\" Stier is a member of a New York Synagogue that has donated food for over a decade. He is outraged that the DHS' response to his demand to know why the practice can"
                        },
                        {
                            "title": "New York City bans food donations - WND", 
                            "text": "New York City bans food donations - WND ban on donated food. In fact, it thrives because of food donations. New York City Rescue Mission has been providing food, clothing, shelter and spiritual hope for needy New Yorkers since 1872. \"We feed over 500 people a day, all through donations,\" said James Varnhagen, NYCRM director. \"Boxed food, canned food, prepared food, we take any food,\" he told WND. \"We couldn't survive without donations,\" he said."
                        }
                    ]
                },
                {
                    "question": "What's the difference between Shia vs. Sunni Islam?",
                    "answer": "The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership [1]. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. [1][2]. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad's son-in-law Ali [2]. Nowadays, Sunni and Shia are the major branches of Islam [3].",
                    "docs": [
                        {
                            "title": "The Sunni vs Shia Divide - Explained - Globaloi", 
                            "text": "centuries-long strained relationship between Sunnis and Shias. As a scholar of Islam and a public educator, I often field questions about Sunnis, Shias and the sects of Islam. What exactly is the Shia-Sunni divide? And what is its history? History of divide Both Sunnis and Shias \u2013 drawing their faith and practice from the Qur\u2019an and the life of the Prophet Muhammad \u2013 agree on most of the fundamentals of Islam. The differences are related more to historical events, ideological heritage and issues of leadership. The first and central difference emerged after the death of Prophet Muhammad in A.D. 632."
                        },
                        {
                            "title": "What\u2019s the difference between Sunni and Shia Islam? \u2013 Macrosnaps", 
                            "text": "What\u2019s the difference between Sunni and Shia Islam? Sunni and Shia identities (the 2 main branches of Islam) first formed around a dispute over leadership succession after the death of the Prophet Muhammad in 632 A.D. Sunni is the larger branch (estimated 85-90% of total world Muslim population) and it's adherents are referred to as \"people of the tradition of Muhammad\", while Shia are \"followers\" of Muhammad's son-in-law and cousin Ali. Sunnis rely heavily on the practice of the Prophet Muhammad and his teachings, the Shia view their ayatollahs as reflections of God on earth. What challenges does the anti-IS"
                        },
                        {
                            "title": "Difference between Sunni and Shia Muslims | Sunni vs Shia Muslims", 
                            "text": "of Muhammad, the last prophet of God. A follower of Islam is known as a Muslim. Many Muslims believe that their sole purpose is to worship and serve God, for which they have established five pillars of Islam that guides a Muslim on almost every aspect of life and society. Due to differences, Muslims have been divided into two primary sects: The Sunnis and the Shias. These two sects have many similarities and both consider themselves are Muslims, following the will of God. However, they are also different from each other in certain aspects. Both the Sunnis and the Shias,"
                        },
                        {
                            "title": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange", 
                            "text": "What is the difference between Shia and Sunni Islam? - Islam Stack Exchange between Mutah marriage and Misyar marriage? What theological and historical factors distinguish Ibadi Islam from either Shia or Sunni schools? What are the principle/fundamental differences between Sunni and Shia? Nikah between a Sunni girl and Shia boy What is the difference between \u201cMubtalat-of-Wudu\u201d of Shia and Sunni? How can the Hadith be reliable when Sunnis and Shia follow different points of reference? Rejection of Mutawatir Hadith in Sunni Islam and Shia Islam"
                        },
                        {
                            "title": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel", 
                            "text": "What is the difference between Sunni and Shia Islam? | Patrick Syder Travel What is the difference between Sunni and Shia Islam? This Channel 4 link answers some of the key questions about the difference between Sunni and Shia Islam and alarmingly, the politics on what is happening and why, in Syria\u2026\u2026. http://www.channel4.com/news/sunni-shia-islam-muslim-syria-middle-east-key-questions \u2190 Ethiopia Appeal \u2013 Help sponsor a nurse to train and to help others G\u00f6bekli Tepe, Turkey: a new wonder of the ancient world by Jeremy Seal (Telegraph Travel Section 23/04/2013) \u2192"
                        }
                    ]
                },
                {
                    "question": "What causes Bi-polar disorder?",
                    "answer": "Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression [1][3]. The spectrum of mood swing may span from days to months [1][2]. We are still not certain of the exact factors that cause such disorder, but genetics is considered a major factor [2][3].",
                    "docs": [
                        {
                            "title": "Bi-polar disorder | definition of Bi-polar disorder by Medical dictionary", 
                            "text": "bi-polar disorder | definition of bi-polar disorder by medical dictionary https://medical-dictionary.thefreedictionary.com/bi-polar+disorder (redirected from bi-polar disorder) related to bi-polar disorder: depression bipolar disorder, formerly known as manic depression, is a mood disorder that causes radical emotional changes and mood swings, from manic, restless highs to depressive, listless lows. most bipolar individuals experience alternating episodes of mania and depression. bipolar disorder is characterized by alternating manic episodes in which the individual feels abnormally euphoric, optimistic, and energetic and depressive periods in which the individual feels sad, hopeless, guilty, and sometimes suicidal. manic or depressive periods may last for days, weeks, or months"
                        },
                        {
                            "title": "Mania and Bi-Polar", 
                            "text": "can go from depressed to \u201csuper happy\u201d all in one day, or even in a few days, does not have a bi-polar disorder Bi-polar looks different depending on the severity of the symptoms. Most bi-polar diagnoses that are made are for bi-polar 2, with bi-polar 1 being much more rare. Bi-polar 1 is so severe that the individual will have periods of such agitation, or such reckless and seemingly foolish behavior that they put themselves or those around them in danger. It is not completely clear what causes bi-polar, but genetics seem to have a large role. The biggest factor"
                        },
                        {
                            "title": "Bi-Polar disorder", 
                            "text": "Bi-Polar disorder Bi-polar is generally a cyclic disease where individuals display depressive and elevated episodes at regular intervals. It is a disorder resulting from the imbalance of the chemicals in the brain that causes a lot of fluctuations of mood. It is a fact that we all experience happy and sad moods, but people with bi-polar disorder experience the changes in mood at an increased level. The cause of this disorder is not known completely. However, it is estimated that there are different factors responsible for it. It is often connected to a genetic component. People suffering from the Bi-polar disorder are"
                        },
                        {
                            "title": "For Individuals \u2014 Adam Schwartz", 
                            "text": "For Individuals \u2014 Adam Schwartz The information is extensive and covers a huge range of topics. Some of the topics include the different types of bi-polar, what it feels like, signs and symptoms, treatments and more. Black Dog Institute bi-polar causes resource specifically covers the variety of areas that could potentially be a cause of bi-polar disorder. Including genetics, environmental factors, pregnancy, and more. Black Dog Institute bi-polar treatments resource specifically covers multiple potential treatments options for bi-polar. Including management, types of psychological treatment, lifestyle changes, and more. Black Dog Institute bi-polar self-test resource is a short self-test for people who may be concerned if"
                        },
                        {
                            "title": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression", 
                            "text": "Depression Bi-polar Disorder Symptoms 2019 | Win Over Depression signs and symptoms of bipolar disorder. Learn more about the common symptoms of bipolar depression that some patients may experience. Home \u00bb Trending Health News \u00bb 10 Warning Signs of Bipolar Disorder: Depression. One of the most serious symptoms of bipolar disorder is. Bi Polar Depression. SEVERE SWINGS What is bipolar disorder, is it the same as manic depression, what are the symptoms and is there a cure? Bipolar disorder, or manic depression, causes symptoms of mania and depression. Read about bipolar disorder treatment, medications, and causes of this. Learn more about the different types of bipolar disorder. Find out"
                        }
                    ]
                },
                {
                    "question": "How do student loans affect getting a mortgage?",
                    "answer": "When applying for a mortgage, student loans can affect the debt to income ratio, which is a key factor in determining the amount that an individual can afford to pay for the mortgage [1]. While student loan repayments do not appear in an individual's credit history and do not affect credit scores, lenders do consider the amount of an individual's student loan repayments when assessing their mortgage application [1][2][3]. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors [2]. It is important to note that student loans do not prevent an individual from getting a mortgage [1].",
                    "docs": [
                        {
                            "title": "Student Loans \u2013 How do they work? | The Financial Review", 
                            "text": "typical debt. Student loan repayments do not appear in an individual\u2019s credit history, therefore there are no implications whatsoever. This also extends to applications for credit cards \u2013 student \u2018loans\u2019 are not acknowledged. One noteworthy aspect that is affected by student loans however, is mortgage applications. Nevertheless, it does not prevent an individual from getting a mortgage. For example, lenders will consider the amount of an individual\u2019s student loan repayments in order to assess the debt to income ratio and therefore establish the amount that the individual can afford to pay for the mortgage. Just as they do with other"
                        },
                        {
                            "title": "How Does Student Loan Debt Affect Buying a Home? | Experian", 
                            "text": "Rates & Affordability How Student Loans Affect Getting a Mortgage Student Loan Impact on Credit Scores Other Factors for Getting Approved for a Mortgage If you're a recent college grad and hope to become a homeowner in the near future, you should know that student loan debt could affect buying a home by making it more difficult to get a mortgage. Some 83% of non-homeowners say student loan debt is preventing them from buying a home, according to the National Association of Realtors (NAR). But while student loan payments can make it harder to save for a down payment on"
                        },
                        {
                            "title": "Studentloanify - How your student loans affect your home mortgage prospects", 
                            "text": "Though it may not seem fair, your student loan situation impacts your home mortgage outlook. Many people carry student loan debt, but it\u2019s the amount of the loan and how you handle your student loan repayment plan that will influence your ability to get a home mortgage as well as what your interest rate will be. Here are some specific factors about your student loan that will affect your home mortgage prospects. On your mortgage loan application, you will have to report how much your monthly student loan payment is. This amount will be deducted from your monthly gross income"
                        },
                        {
                            "title": "How do student loans affect your credit score? | Student Loan Planner", 
                            "text": "How do student loans affect your credit score? | Student Loan Planner Your credit score is the three-digit number that dictates a lot in your adult life. Whether you\u2019re applying for a mortgage or looking to get an auto loan, this seemingly arbitrary number determines whether you get approved for a loan and also affects your interest rate. If you\u2019re a student loan borrower you may wonder, \u201cDo student loans affect credit score?\u201d You might be especially curious if you\u2019re in the process of applying for a mortgage. Here\u2019s how student loans affect your credit score and what to know for big life events, like getting a mortgage. Do student loans affect"
                        },
                        {
                            "title": "Does Student Loan Debt Affect Getting A Mortgage?", 
                            "text": "Does Student Loan Debt Affect Getting A Mortgage? Home \u00bb Does Student Loan Debt Affect Getting A Mortgage? Last year, I helped answer a reader\u2019s question about applying for a mortgage while on Income Based Repayment. However, over the last several months, I\u2019ve been getting bombarded with questions about how student loan debt impacts your ability to get a mortgage. Maybe it\u2019s because the housing market is improving, or maybe it\u2019s because people are finally taking their student loan debt seriously. Anyway, I wanted to share a few reader questions and then look at whether student loan debt affects getting a mortgage. Here are the reader questions I\u2019ve"
                        }
                    ]
                }
            ],
            
            'hagrid': [
                {
                    "question": "Where did Jehovah's Witnesses originate?",
                    "answer": "Jehovah's Witnesses originated as a branch of the Bible Student movement, which developed in the United States in the 1870s among followers of Christian Restorationist minister Charles Taze Russell [2]. The movement began with Bible Student missionaries being sent to England in 1881, and the first overseas branch was opened in London in 1900 [2]. After Russell's death in 1916, the movement split into several organizations, with one led by his successor Joseph \"Judge\" Rutherford retaining control of the Watch Tower Society and eventually becoming known as Jehovah's Witnesses [2].",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "Jehovah's Witnesses were an outgrowth of the International Bible Students, who began missionary work in Europe in the 1890s. A German branch office of the Watch Tower Society opened in Elberfeld in 1902. By 1933, almost 20,000 Witnesses were counted as active door-to-door preachers, and their annual Memorial service was attracting almost 25,000 people. In Dresden, there were more Bible Students than in New York, where the Watch Tower Society was headquartered."
                    },
                    {
                        "title": "Document 2",
                        "text": "Jehovah's Witnesses originated as a branch of the Bible Student movement, which developed in the United States in the 1870s among followers of Christian Restorationist minister Charles Taze Russell. Bible Student missionaries were sent to England in 1881 and the first overseas branch was opened in London in 1900. The group took on the name International Bible Students Association and by 1914 it was also active in Canada, Germany, Australia and other countries. The movement split into several rival organizations after Russell's death in 1916, with one—led by Russell's successor, Joseph \"Judge\" Rutherford—retaining control of both his magazine, \"The Watch Tower\", and his legal and publishing corporation, the Watch Tower Bible and Tract Society of Pennsylvania."
                    }
                    ]
                },
                {
                    "question": "What is a conurbation?",
                    "answer": "A conurbation is a region comprising a number of cities, large towns, and other urban areas that have merged through population growth and physical expansion to form one continuous urban or industrially developed area [1]. The term is commonly used in Britain, while in the United States, polycentric metropolitan areas may have their own designations such as San Francisco Bay Area or Dallas-Fort Worth Metroplex [2]. The New York metropolitan area serves as an example, encompassing 30 counties across four states with an estimated population of 21,961,994 in 2007 [3]. The term was coined in 1915 by Patrick Geddes in his book \"Cities In Evolution\" [4].",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "A conurbation is a region comprising a number of cities, large towns, and other urban areas that, through population growth and physical expansion, have merged to form one continuous urban or industrially developed area. In most cases, a conurbation is a polycentric urbanised area, in which transportation has developed to link areas to create a single urban labour market or travel to work area."
                    },
                    {
                        "title": "Document 2",
                        "text": "The term as described is used in Britain, whereas in the United States each polycentric \"metropolitan area\" may have its own common designation, such as San Francisco Bay Area or the Dallas-Fort Worth Metroplex. Conurbation consists of adjacent metropolitan areas that are connected with one another by urbanization Internationally, the term \"urban agglomeration\" is often used to convey a similar meaning to \"conurbation.\" A conurbation should also be contrasted with a megalopolis, where the urban areas are close but not physically contiguous and where the merging of labour markets has not yet developed."
                    },
                    {
                        "title": "Document 3",
                        "text": "One example of a conurbation is the expansive concept of the New York metropolitan area (the Tri-state region) centered on New York City, including 30 counties spread among New York State, New Jersey, Connecticut, and Pennsylvania, with an estimated population of 21,961,994 in 2007. Approximately one-fifteenth of all U.S. residents live in the Greater New York City area. This conurbation is the result of several central cities whose urban areas have merged."
                    },
                    {
                        "title": "Document 4",
                        "text": "The term \"conurbation\" was coined in 1915 by Patrick Geddes in his book \"Cities In Evolution\". He drew attention to the ability of the then new technology of electric power and motorised transport to allow cities to spread and agglomerate together, and gave as examples \"Midlandton\" in England, the Ruhr in Germany, Randstad in the Netherlands and North Jersey in the United States."
                    },
                    {
                        "title": "Document 5",
                        "text": "Although the exact boundaries of any conurbation are open to debate, dependent on what criteria are used to determine where an urban area ceases, the Office for National Statistics defines the West Midlands Built Up Area as including the urban areas of Birmingham, Wolverhampton, Solihull, West Bromwich, Dudley, and Walsall amongst others. These settlements are not coterminous with the Metropolitan Boroughs of the same name."
                    }
                    ]
                },
                {
                    "question": "When did the movie Scott Pilgrim come out?",
                    "answer": "The movie Scott Pilgrim vs. the World received a wide release in North America on August 13, 2010 [2]. The film premiered at the San Diego Comic-Con International on July 22, 2010, following a panel discussion [2]. Additional screenings were held at the Fantasia Festival in Montreal on July 27, 2010, and at Movie-Con III in London on August 15, 2010 [1].",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "A \"Scott Pilgrim vs. the World\" panel featured at the San Diego Comic-Con International held on July 22, 2010. After the panel, Wright invited selected members of the audience for a screening of the film which was followed by a performance by Metric. \"Scott Pilgrim\" was also shown at the Fantasia Festival in Montreal, Quebec, Canada on July 27, 2010 and was also featured at the Movie-Con III in London, England on August 15, 2010."
                    },
                    {
                        "title": "Document 2",
                        "text": "A film adaptation of the comics was proposed following the release of the first volume. Wright became attached to the project, and filming began in March 2009 in Toronto. The film premiered after a panel discussion at the San Diego Comic-Con International on July 22, 2010, and received a wide release in North America on August 13, 2010."
                    }
                    ]
                }
            ],
            
            'msmarco': [
                {
                    "question": "why is albumin normally absent in urine",
                    "answer": "The kidney cells generally prevent large molecules including proteins, from being excreted [1]",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "Albumin is a protein present in the blood. Proteins are normally absent in urine because the kidney cells generally prevent large molecules including proteins, from being excreted. Albumin in your urine may be an early sign of kidney problems although some people have small amounts of albumin in their urine normally."
                    },
                    {
                        "title": "Document 2",
                        "text": "A normal amount of albumin in your urine is less that 20 mg a day. A normal total protein amount in your urine is less that 150 mg a day. If your test shows high levels of urine albumin, or a rise in urine albumin, it could mean you have kidney damage or disease."
                    },
                    {
                        "title": "Document 3",
                        "text": "Guide. An albumin test checks urine for a protein called albumin. Albumin is normally found in the blood and filtered by the kidneys. When the kidneys are working as they should, there may be a very small amount of albumin in the urine. But when the kidneys are damaged, abnormal amounts of albumin leak into the urine. This is called albuminuria."
                    },
                    {
                        "title": "Document 4",
                        "text": "Share. Albumin is a protein present in the blood. Proteins are normally absent in urine because the kidney cells generally prevent large molecules including proteins, from being excreted. Some proteins may appear in the urine in normal individuals also if blood levels are very high. In kidney diseases, albumin will appear in the urine even with normal blood levels."
                    },
                    {
                        "title": "Document 5",
                        "text": "Albumin is a protein present in the blood. Proteins are normally absent in urine because the kidney cells generally prevent large molecules including proteins, from being excreted. Some proteins may appear in the urine in normal individuals also if blood levels are very high. In kidney diseases, albumin will appear in the urine even with normal blood levels."
                    }
                    ]
                },
                {
                    "question": "walgreens store sales average",
                    "answer": "Approximately $15,000 per year [1]",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed."
                    },
                    {
                        "title": "Document 2",
                        "text": "The average revenue in 2011 of a Starbuck Store was $1,078,000, up  from $1,011,000 in 2010.    The average ticket (total purchase) at domestic Starbuck stores in  No … vember 2007 was reported at $6.36.    In 2008, the average ticket was flat (0.0% change)."
                    },
                    {
                        "title": "Document 3",
                        "text": "In fiscal 2014, Walgreens opened a total of 184 new locations and acquired 84 locations, for a net decrease of 273 after relocations and closings. How big are your stores? The average size for a typical Walgreens is about 14,500 square feet and the sales floor averages about 11,000 square feet. How do we select locations for new stores? There are several factors that Walgreens takes into account, such as major intersections, traffic patterns, demographics and locations near hospitals."
                    },
                    {
                        "title": "Document 4",
                        "text": "th store in 1984, reaching $4 billion in sales in 1987, and $5 billion two years later. Walgreens ended the 1980s with 1,484 stores, $5.3 billion in revenues and $154 million in profits. However, profit margins remained just below 3 percent of sales, and returns on assets of less than 10 percent."
                    },
                    {
                        "title": "Document 5",
                        "text": "The number of Walgreen stores has risen from 5,000 in 2005 to more than 8,000 at present. The average square footage per store stood at approximately 10,200 and we forecast the figure to remain constant over our review period. Walgreen earned $303 as average front-end revenue per store square foot in 2012."
                    },
                    {
                        "title": "Document 6",
                        "text": "Your Walgreens Store. Select a store from the search results to make it Your Walgreens Store and save time getting what you need. Your Walgreens Store will be the default location for picking up prescriptions, photos, in store orders and finding deals in the Weekly Ad."
                    }
                    ]
                },
                {
                    "question": "cost to frame basement",
                    "answer": "$2.51 - $3.17 per square foot [6]",
                    "docs": [
                    {
                        "title": "Document 1",
                        "text": "1 A lot depends on how much is included and the quality of the final project. 2  Finishing a basement in a relatively new home can start around $20 -$35 a square foot, or $30,000 -$70,000 for 1,500-2,000 square feet -- but it can cost $100,000 or more to create a fully-finished space."
                    },
                    {
                        "title": "Document 2",
                        "text": "The cost of a basement is between 10 and 35 dollars per square feet. Let's say an average basement is 1,000 square feet of finished space. So the cost of a basement is between $10,000 and $35,000. 10k if you're doing most of the work yourself and up to $35,000 if you're hiring a contractor to finish your basement. Now."
                    },
                    {
                        "title": "Document 3",
                        "text": "There are some options when it comes to finishing a basement, and they all come with a cost. If you choose a basic do-it-yourself makeover, including adding moisture control, framing in walls and adding decor and furnishings, your cost will run between $10,000 and $27,000."
                    },
                    {
                        "title": "Document 4",
                        "text": "Expect this type of basement to cost more at least $65,000. A basement finishing system is another option, running $45,000 to $55,000. These out-of-the-box systems include everything from moisture-proof, insulated wall panels to ceilings to lighting."
                    },
                    {
                        "title": "Document 5",
                        "text": "1 Adding a basement to an existing house can cost $30,000-$70,000 or more, depending on the size of the existing crawlspace or half-basement or if the house is on a slab foundation, and how much of the work is do-it-yourself and how much is done by contractors."
                    },
                    {
                        "title": "Document 6",
                        "text": "Our free calculator uses recent, trusted data to estimate costs for your Basement Wall Framing project. For a basic 125 square feet project in zip code 47474, the benchmark cost to Frame Basement Walls ranges between $2.51 - $3.17 per square foot* . To estimate costs for your project:"
                    }
                    ]
                }
            ],
            
            'qampari': [
                {
                    'question': "Which books were written by Nevil Shute?",
                    'answer': "Marazan [1], Stephen Morris [1], Beyond the Black Stump [2], Lonely Road [2], The Chequer Board [2], In the Wet [2], Trustee from the Toolroom [2], Round the Bend [2], No Highway [3], Ruined City [3], On the Beach [3].",
                    'docs': [
                        {"title": "Nevil Shute", "text": "early stages. My congratulations.\" His celebrity as a writer caused the Ministry of Information to send him to the Normandy Landings on 6 June 1944 and later to Burma as a correspondent. He finished the war with the rank of lieutenant commander in the Royal Navy Volunteer Reserves (RNVR). Shute's first novel, \"Stephen Morris\", was written in 1923, but not published until 1961. His first published novel was \"Marazan\", which came out in 1926. After that he averaged one novel every two years through the 1950s, with the exception of a six-year hiatus while he was establishing his own aircraft"},
                        {"title": "Nevil Shute", "text": "theme is the bridging of social barriers such as class (\"Lonely Road\" and \"Landfall\"), race (\"The Chequer Board\"), or religion (\"Round the Bend\"). The Australian novels are individual hymns to that country, with subtle disparagement of the mores of the United States (\"Beyond the Black Stump\") and overt antipathy towards the post-World War II socialist government of Shute's native Britain (\"The Far Country\" and \"In the Wet\"). Shute's heroes tended to be like himself: middle class solicitors, doctors, accountants, bank managers, engineers, generally university graduates. However (as in \"Trustee from the Toolroom\"), Shute valued the honest artisans and their social"},
                        {"title": "Nevil Shute", "text": "construction company, Airspeed Ltd. His popularity grew slowly with each novel, but he became much more famous after the publication of \"On the Beach\" in 1957. Shute's novels are written in a simple, highly readable style, with clearly delineated plot lines. Where there is a romantic element, sex is referred to only obliquely. Many of the stories are introduced by a narrator who is not a character in the story. The most common theme in Shute's novels is the dignity of work, spanning all classes, whether an Eastern European bar \"hostess\" (\"Ruined City\") or brilliant boffin (\"No Highway\"). Another recurrent"},
                        {"title": "The Chequer Board", "text": "the Burmese people\", both of which are central to the book's story. Shute was concerned that sales of the book in the United States would be negatively impacted by the book's open-minded handling of racial issues; as it turned out, sales soared. Shute and his wife traveled the U.S. on Greyhound buses to \"\"get in touch with the man on the street,\"\" finding the experience refreshing. Afterwards he wrote \"\"Sincerity is the first attribute for making money in the business of writing novels.\"\" The Chequer Board The Chequer Board is a novel by Nevil Shute, first published in the United"},
                        {"title": "In the Wet", "text": "had used the idea of multiple votes for merit in his short story \"The Curious Republic of Gondour\". In the Wet In The Wet is a novel by Nevil Shute that was first published in the United Kingdom in 1953. It contains many of the typical elements of a hearty and adventurous Shute yarn such as flying, the future, mystic states, and ordinary people doing extraordinary things. The story is opened by its initial narrator – an Anglican priest in the Bush Brotherhood named Roger Hargreaves – who describes his ordinary circumstances in a large parish of the Australian outback"}
                    ]
                },
                {
                    'question': "Which film has Gong Li as a member of its cast?",
                    'answer': "The Story of Qiu Ju [1], Farewell My Concubine [2], Flirting Scholar [2], The Monkey King 2 [3], Mulan [3], Saturday Fiction [3], Coming Home [3].",
                    'docs': [
                        {"title": "Gong Li", "text": "Gong Li Gong Li (born 31 December 1965) is a Chinese-born Singaporean film actress. She achieved international prominence through her close collaborations with Chinese director Zhang Yimou and won the Volpi Cup for Best Actress at Venice for her performance in his 1992 film \"The Story of Qiu Ju\". She has been credited with helping to bring Chinese cinema to prominence in Europe and the United States. In 2006, she was voted the most beautiful woman in China. Gong has won numerous accolades for her work as an actress; she won the New York Film Critics Circle Award for Best"},
                        {"title": "Gong Li", "text": "making her realize that she has assisted the dark cynical system. In 1993, she received a New York Film Critics Circle award for her role in \"Farewell My Concubine\" (1993). Directed by Chen Kaige, the film was her first major role with a director other than Zhang Yimou. In the same year, she was awarded with the Berlinale Camera at the 43rd Berlin International Film Festival. \"Premiere\" magazine ranked her performance in \"Farewell My Concubine\" as the 89th greatest performance of all time. She also worked with renowned director Stephen Chow in comedy films \"\" (1991) and \"Flirting Scholar\" (1993)."},
                        {"title": "Gong Li", "text": "International Film Festival. Later that same year, she reunited with Zhang Yimou for the film \"Coming Home\", which is set during the throes of the Cultural Revolution; this film was their first collaboration since 2006. In 2016, Gong took on her first action role in \"The Monkey King 2\", playing the White Bone Demon. In 2018, Gong was cast in Lou Ye's period drama \"Saturday Fiction\", where she plays an actress who is working undercover gathering intelligence for the Allies. That year, she was also cast in the live-action adaptation of the 1998 Disney animated film \"Mulan\", as an unspecified"},
                        {"title": "Zhang Yimou", "text": "in Zhang's earlier films. \"Raise the Red Lantern\" was nominated in the Best Foreign Language Film category at the 1992 Academy Awards, becoming the second Chinese film to earn this distinction (after Zhang's \"Ju Dou\"). It eventually lost out to Gabriele Salvatores's \"Mediterraneo\". Zhang's next directorial work, \"The Story of Qiu Ju\", in 1992, once again starring Gong Li in the lead role. The film, which tells the tale of a peasant woman seeking justice for her husband after he was beaten by a village official, was a hit at film festivals and won the Golden Lion award at the"},
                        {"title": "Gong Li", "text": "Gong Li Gong Li (born 31 December 1965) is a Chinese-born Singaporean film actress. She achieved international prominence through her close collaborations with Chinese director Zhang Yimou and won the Volpi Cup for Best Actress at Venice for her performance in his 1992 film \"The Story of Qiu Ju\". She has been credited with helping to bring Chinese cinema to prominence in Europe and the United States. In 2006, she was voted the most beautiful woman in China. Gong has won numerous accolades for her work as an actress; she won the New York Film Critics Circle Award for Best"}
                    ]
                },
                {
                    'question': "In which years did Patti LaBelle publish music?",
                    'answer': "2006 [1], 1977 [2], 2004 [3], 2005 [3], 2000 [3], 2006 [3].",
                    'docs': [
                        {"title": "The Gospel According to Patti LaBelle", "text": "The Gospel According to Patti LaBelle The Gospel According to Patti LaBelle is the first gospel album released by singer Patti LaBelle, released in November 2006. This project began three years ago when Patti's late musical director and close friend Budd Ellison told a skeptical LaBelle that \"it's now or never, Patti.\" The album is dedicated to his memory as he succumbed to prostate cancer before the album saw a release. The album was released on November 21, 2006 through indie label Umbrella/Bungalow Records, also home to Carl Thomas, Rodney Jerkins, Dean \"DC\" Charles, and other artists. \"The Gospel According"},
                        {"title": "Patti LaBelle (album)", "text": "scaled the high sixties on the \"Billboard\" R&B chart, it soon became one of her famous show-stoppers while performing the song. LaBelle performed the song at her first solo concert in London, getting a standing ovation, which helped to give LaBelle motivation to continue her career. The album, when released, performed successfully, reaching number 62 on the \"Billboard\" 200 and number 31 on the R&B albums chart, while critics hailed the album. Patti LaBelle (album) Patti LaBelle is the debut solo album by singer Patti LaBelle, released in 1977. The first album LaBelle recorded after sixteen years fronting the band"},
                        {"title": "Patti LaBelle", "text": "win. In 2000, LaBelle released her final MCA album, \"When a Woman Loves\", before signing with Def Soul Classics to release the 2004 album, \"Timeless Journey\". Following the release of her 2005 covers album, \"Classic Moments\", LaBelle engaged in a rivalry with Antonio \"L.A.\" Reid over the direction of her career, leading to her leaving the label.In the same year, the World Music Awards recognized her years in the music business by awarding her the Legend Award. In 2006, she released her first gospel album, \"The Gospel According to Patti LaBelle\" on the Bungalo label, the album later peaking at"},
                        {"title": "Patti LaBelle", "text": "Patti LaBelle Patti LaBelle (born Patricia Louise Holt; May 24, 1944) is an American singer, actress, and entrepreneur. LaBelle began her career in the early 1960s as lead singer and front woman of the vocal group, Patti LaBelle and the Bluebelles. Following the group's name change to Labelle in the early 1970s, they released the iconic disco song \"Lady Marmalade\" and the group later became the first African-American vocal group to land the cover of \"Rolling Stone\" magazine. After the group split in 1976, LaBelle began a successful solo career, starting with her critically acclaimed debut album, which included the"},
                        {"title": "The Gospel According to Patti LaBelle", "text": "Billboard's Top Gospel Albums chart for 17 weeks. \"Where Love Begins,\" a duet with Yolanda Adams was played frequently on R&B and gospel radio stations and debuted at #68 on Billboard's Hot R&B/Hip-Hop tracks. The second single \"Anything\" featuring Kanye West, Mary Mary and Consequence hit #64 on Billboards Hot R&B/Hip-Hop tracks. In 2008, the album was nominated for a Dove Award for Contemporary Gospel Album of the Year at the 39th GMA Dove Awards. The Gospel According to Patti LaBelle The Gospel According to Patti LaBelle is the first gospel album released by singer Patti LaBelle, released in November"}
                    ]
                },
            ],
            
            
            'natural_questions': [
                {
                    "question": "who sings does he love me with reba",
                    "answer": "Linda Davis [1]",
                    "docs": [
                    {
                        "title": "Does He Love You",
                        "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members"
                    },
                    {
                        "title": "Red Sandy Spika dress of Reba McEntire",
                        "text": "Red Sandy Spika dress of Reba McEntire American recording artist Reba McEntire wore a sheer red dress to the 1993 Country Music Association Awards ceremony on September 29, 1993. The sheer fabric was covered with sequins, and cut with a low neckline. The garment was designed by stylist Sandy Spika, and McEntire wore it during a duet performance of \"Does He Love You\" with Linda Davis. McEntire later said, \"I got more press off that dress than if I'd won entertainer of the year.\" According to McEntire, when her little sister, Susie, saw her on stage she leaned over and"
                    },
                    {
                        "title": "Greatest Hits Volume Two (Reba McEntire album)",
                        "text": "(then a background singer in McEntire's road band), was the first single and turned out to be a smash. It reached number 1 on the country charts. The song also earned them a Grammy award for Best Country Vocal Collaboration as well as the CMA Award for \"Vocal Event of the Year\". CMT ranked the song at No. 9 on their list of 100 Greatest Duets. \"Does He Love You\" is the first of three duets featuring Reba and Linda Davis. The album's other new track was \"They Asked About You\", which peaked at No. 7 on the country chart."
                    },
                    {
                        "title": "Shoot for the Moon (album)",
                        "text": "Shoot for the Moon (album) Shoot for the Moon is the third album by country music artist Linda Davis, It was her first to achieve placement on the Billboard Music Charts. It was the first album released following a win at the 1993 Grammy Awards for Best Country Vocal Collaboration (with country superstar Reba McEntire) for their hit \"Does He Love You.\" The album rose to the number 28 position on the Country Albums chart, and two of its tracks were relatively minor hits on the singles charts: \"Company Time\" at number 43, and \"Love Didn't Do It\" at number"
                    }
                    ]
                },
                {
                    "question": "where do the great lakes meet the ocean",
                    "answer": "the Saint Lawrence River [1]",
                    "docs": [
                    {
                        "title": "Great Lakes",
                        "text": "Great Lakes The Great Lakes (), also called the Laurentian Great Lakes and the Great Lakes of North America, are a series of interconnected freshwater lakes located primarily in the upper mid-east region of North America, on the Canada–United States border, which connect to the Atlantic Ocean through the Saint Lawrence River. They consist of Lakes Superior, Michigan, Huron, Erie, and Ontario, although hydrologically, there are four lakes, Superior, Erie, Ontario, and Michigan-Huron. The lakes are interconnected by the Great Lakes Waterway. The Great Lakes are the largest group of freshwater lakes on Earth by total area, and second largest"
                    },
                    {
                        "title": "Great Lakes",
                        "text": "its impact on the environment. On December 18, 2006, the Coast Guard announced its decision to withdraw the entire proposal. Officials said they would look into alternative ammunition, modifying the proposed zones and have more public dialogue before proposing a new plan. Dynamically updated data Great Lakes The Great Lakes (), also called the Laurentian Great Lakes and the Great Lakes of North America, are a series of interconnected freshwater lakes located primarily in the upper mid-east region of North America, on the Canada–United States border, which connect to the Atlantic Ocean through the Saint Lawrence River. They consist of"
                    },
                    {
                        "title": "Great Lakes",
                        "text": "form a single, naturally interconnected body of fresh water, within the Great Lakes Basin. They form a chain connecting the east-central interior of North America to the Atlantic Ocean. From the interior to the outlet at the Saint Lawrence River, water flows from Superior to Huron and Michigan, southward to Erie, and finally northward to Lake Ontario. The lakes drain a large watershed via many rivers, and are studded with approximately 35,000 islands. There are also several thousand smaller lakes, often called \"inland lakes,\" within the basin. The surface area of the five primary lakes combined is roughly equal to"
                    },
                    {
                        "title": "Lake Ontario",
                        "text": "the Niagara River from Lake Erie. The last in the Great Lakes chain, Lake Ontario serves as the outlet to the Atlantic Ocean via the Saint Lawrence River. It is the only Great Lake not to border the state of Michigan. Lake Ontario is the easternmost of the Great Lakes and the smallest in surface area (7,340 sq mi, 18,960 km), although it exceeds Lake Erie in volume (393 cu mi, 1,639 km). It is the 14th largest lake in the world. When its islands are included, the lake's shoreline is long. As the last lake in the Great Lakes'"
                    },
                    {
                        "title": "New York (state)",
                        "text": "that includes within its borders parts of the Great Lakes and the Atlantic Ocean. The Hudson River begins near Lake Tear of the Clouds and flows south through the eastern part of the state, without draining Lakes George or Champlain. Lake George empties at its north end into Lake Champlain, whose northern end extends into Canada, where it drains into the Richelieu River and then ultimately the Saint Lawrence River. The western section of the state is drained by the Allegheny River and rivers of the Susquehanna and Delaware River systems. Niagara Falls is shared between New York and Ontario"
                    }
                    ]
                },
                {
                    "question": "when does the new my hero academia movie come out",
                    "answer": "July 5, 2018 [1]",
                    "docs": [
                    {
                        "title": "My Hero Academia: Two Heroes",
                        "text": "would be joining the cast as Melissa Shield and Katsuhisa Namase would play David Shield, both original characters. On June 11, 2018, \"Weekly Shōnen Jump\" announced that Rikiya Koyama had been cast as the film's villain, Wolfram. Masaki Suda performs the film's theme song , which was written and composed by Hiromu Akita of amazarashi. Funimation and Toho premiered the film at Anime Expo in Los Angeles on July 5, 2018, and it was later released in Japan on August 3 of that year. The first one million audience members to see the movie will receive a special book containing"
                    },
                    {
                        "title": "My Hero Academia",
                        "text": "announced in the 44th issue of \"Weekly Shōnen Jump\" magazine of 2018. This was later confirmed with the airing of the final episode to season three on September 29, 2018. On December 19, 2018, the \"My Hero Academia\" website confirmed a release date of October 2019, along with a key visual. An anime film was announced in December 2017 and features an original story set after the manga's \"Final Exam\" arc. Titled , the film had its world premiere at Anime Expo on July 5, 2018, and the Japanese theatrical release began screening on August 3, 2018, with the staff"
                    }
                    ]
                }
            ]
        }
    
    def get_prompt(self, 
                   dataset_name: str, 
                   method: MethodType, 
                   question: str, 
                   context: str,
                   include_few_shot: bool = True,
                   num_few_shot: int = 1) -> str:
        """Generate a simplified prompt following ALCE approach"""
        
        if method not in [MethodType.POST_RETRIEVAL, MethodType.POST_GENERATION, MethodType.POST_GENERATION_LLM_SHORT, MethodType.POST_GENERATION_LLM_LONG]:
            raise ValueError("This simplified version only supports post-retrieval, post-generation, post-generation-llm-short, and post-generation-llm-long methods")
        
        # Get dataset-specific instruction
        instruction = self.dataset_instructions.get(dataset_name)
        
        if method == MethodType.POST_RETRIEVAL:
            # Handle post-retrieval method
            prompt_parts = []
            
            # Add few-shot examples if requested and available
            if include_few_shot and dataset_name in self.few_shot_examples:
                examples = self.few_shot_examples[dataset_name]
                # Limit to requested number of examples
                examples_to_use = examples[:num_few_shot]
                
                for example in examples_to_use:
                    # Format the example
                    example_docs = ""
                    for i, doc in enumerate(example['docs'], 1):
                        example_docs += f"Document [{i}](Title: {doc['title']}): {doc['text']}\n"
                    
                    prompt_parts.append(f"{instruction}\n\nQuestion: {example['question']}\n\n{example_docs.strip()}\n\nAnswer: {example['answer']}\n\n")
            
            # Add the main task
            prompt_parts.append(f"{instruction}\n\nQuestion: {question}\n\n{context}\nAnswer: ")
            
            return ''.join(prompt_parts)
        
        elif method == MethodType.POST_GENERATION_LLM_SHORT:
            # Handle post-generation-llm-short method - return dict with initial_prompt and short citation_prompt
            initial_prompt = f"{instruction}\n\nQuestion: {question}\nAnswer: "
            
            # Use short citation prompt for all datasets with few-shot examples
            citation_prompt = self.get_short_citation_prompt(question, "{initial_answer}", context, dataset_name, num_few_shot if include_few_shot else 0)
            
            return {
                'initial_prompt': initial_prompt,
                'citation_prompt': citation_prompt
            }
        
        elif method == MethodType.POST_GENERATION_LLM_LONG:
            # Handle post-generation-llm-long method - return dict with initial_prompt and long citation_prompt
            initial_prompt = f"{instruction}\n\nQuestion: {question}\nAnswer: "
            
            # Use long citation prompt for all datasets with few-shot examples
            citation_prompt = self.get_long_citation_prompt(question, "{initial_answer}", context, dataset_name, num_few_shot if include_few_shot else 0)
            
            return {
                'initial_prompt': initial_prompt,
                'citation_prompt': citation_prompt
            }
    
    def get_clean_instruction(self, dataset_name: str) -> str:
        """Get clean instruction without citation parts for TF-IDF post-generation"""
        if dataset_name in self.clean_instructions:
            return self.clean_instructions[dataset_name]
        else:
            # Fallback to basic instruction
            return "Instruction: Write an accurate, concise answer for the given question."
    
    def get_short_citation_prompt(self, initial_question: str, initial_answer: str, context: str, dataset_name: str = None, num_few_shot: int = 0) -> str:
        """Get a short citation prompt without detailed instructions"""

        if dataset_name == 'asqa':
            instruction = f"""Instruction: Given the above question, answer and documents, add numeric in-text citations, using [1][2][3], to support each claim in the answer."""
        elif dataset_name == 'eli5':
            instruction = f"""Instruction: Given the above question, answer and documents, add numeric in-text citations, using [1][2][3], to support each claim in the answer."""
        elif dataset_name == 'hagrid':
            instruction = f"""Instruction: Given the above question, answer and documents, add numeric in-text citations, using [1][2][3], to support each claim in the answer."""
        elif dataset_name == 'msmarco':
            instruction = f"""Instruction: Given the above question, answer and documents, add numeric in-text citations, using [1][2][3], to support each claim in the answer."""
        elif dataset_name == 'natural_questions':
            instruction = f"""Instruction: Given the above question, answer and documents, add a numeric in-text citation to support the answer using [1], [2], [3], etc."""
        elif dataset_name == 'qampari':
            instruction = f"""Instruction: Given the above question, answer and documents, add numeric in-text citations, using [1], [2], [3], etc. to support each corresponding answer in the list."""
        prompt_parts = []
        
        # Add few-shot examples with citations if requested
        if num_few_shot > 0 and dataset_name and dataset_name in self.few_shot_examples:
            examples = self.few_shot_examples[dataset_name][:num_few_shot]
            clean_examples = self.get_clean_few_shot_examples(dataset_name, num_few_shot)
            
            for idx, example in enumerate(examples):
                # Use the existing few-shot example structure
                prompt_parts.append(f"Question: {example['question']}\n")
                
                # Get the clean answer for this specific example
                if idx < len(clean_examples):
                    clean_answer = clean_examples[idx]['answer']
                else:
                    # Fallback: remove citations from the original answer
                    clean_answer = self._remove_citations(example['answer'])
                
                prompt_parts.append(f"Answer without citations: {clean_answer}\n")
                
                # Add the documents from the existing structure
                if 'docs' in example and example['docs']:
                    prompt_parts.append("Documents:")
                    for doc_idx, doc in enumerate(example['docs']):
                        prompt_parts.extend([
                            f"[{doc_idx+1}] {doc.get('title', 'No title')}",
                            f"{doc.get('text', 'No text')}\n"
                        ])
                prompt_parts.append(f"{instruction}\n")
                prompt_parts.append(f"Answer with citations: {example['answer']}\n")


        
        # Add the main task
        prompt_parts.extend([
            f"Question: {initial_question}",
            "",
            f"Answer without citations: {initial_answer}",
            "",
            f"Documents:",
            f"{context}",
            "",
            f"{instruction}",
            "",
            "Answer with citations: "
        ])        
        return '\n'.join(prompt_parts)
    
    def get_long_citation_prompt(self, initial_question: str, initial_answer: str, context: str, dataset_name: str = None, num_few_shot: int = 0) -> str:
        """Get a long citation prompt with detailed instructions and few-shot examples"""
        # Add instruction based on dataset
        if dataset_name == 'asqa':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer that can be supported by the documents.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Only cite documents that directly support the claim.
4. Add the most relevant citation(s) for each claim.
5. If a claim cannot be supported by any document, leave it without citation.
6. Cite at least one document and at most three documents in each sentence.
7. Use multiple citations only if multiple documents are needed to fully support the claim.
8. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."""

        elif dataset_name == 'eli5':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer that can be supported by the documents.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Only cite documents that directly support the claim.
4. Add the most relevant citation(s) for each claim.
5. If a claim cannot be supported by any document, leave it without citation.
6. Cite at least one document and at most three documents in each sentence.
7. Use multiple citations only if multiple documents are needed to fully support the claim.
8. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."""

        elif dataset_name == 'hagrid':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer that can be supported by the documents.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Only cite documents that directly support the claim.
4. Add the most relevant citation(s) for each claim.
5. If a claim cannot be supported by any document, leave it without citation.
6. Cite at least one document and at most three documents in each sentence.
7. Use multiple citations only if multiple documents are needed to fully support the claim.
8. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."""

        elif dataset_name == 'msmarco':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer that can be supported by the documents.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Only cite documents that directly support the claim.
4. Add the most relevant citation(s) for each claim.
5. If a claim cannot be supported by any document, leave it without citation.
6. Cite at least one document and at most three documents in each sentence.
7. Use multiple citations only if multiple documents are needed to fully support the claim.
8. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."""

        elif dataset_name == 'natural_questions':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add a numeric in-text citation to support the answer using [1], [2], [3], etc.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Add a numeric in-text citation [n] immediately after the claim.
4. Only cite a document that directly support the claim.
5. If multiple documents support the claim, only add the most relevant citation for each claim.
6. If a claim cannot be supported by any document, leave it without citation."""

        elif dataset_name == 'qampari':
            instruction = f"""Instructions:
1. Given the above question, answer and documents, add numeric in-text citations to support each claim in the answer.
2. Keep the original answer text and punctuation intact - DO NOT include the document text in your response or any other content.
3. Add numeric in-text citations, using [1], [2], [3], etc. immediately after each answer that can be supported by a document.
4. Only cite a document that directly supports the corresponding answer in the list."""

        else:
            instruction = "Given the following answer and numbered documents, add numeric in-text citations [1][2][3] etc. to support each claim in the answer."
        
        prompt_parts = []
        
        # Add few-shot examples with citations if requested
        if num_few_shot > 0 and dataset_name and dataset_name in self.few_shot_examples:
            examples = self.few_shot_examples[dataset_name][:num_few_shot]
            clean_examples = self.get_clean_few_shot_examples(dataset_name, num_few_shot)
            
            for idx, example in enumerate(examples):
                # Use the existing few-shot example structure
                prompt_parts.append(f"Question: {example['question']}\n")
                
                # Get the clean answer for this specific example
                if idx < len(clean_examples):
                    clean_answer = clean_examples[idx]['answer']
                else:
                    # Fallback: remove citations from the original answer
                    clean_answer = self._remove_citations(example['answer'])
                
                prompt_parts.append(f"Answer without citations: {clean_answer}\n")
                
                # Add the documents from the existing structure
                if 'docs' in example and example['docs']:
                    prompt_parts.append("Documents:")
                    for doc_idx, doc in enumerate(example['docs']):
                        prompt_parts.extend([
                            f"[{doc_idx+1}] {doc.get('title', 'No title')}",
                            f"{doc.get('text', 'No text')}\n"
                        ])
                prompt_parts.append(f"{instruction}\n")
                prompt_parts.append(f"Answer with citations: {example['answer']}\n")


        
        # Add the main task
        prompt_parts.extend([
            f"Question: {initial_question}",
            "",
            f"Answer without citations: {initial_answer}",
            "",
            f"Documents:",
            f"{context}",
            "",
            f"{instruction}",
            "",
            "Answer with citations: "
        ])        
        return '\n'.join(prompt_parts)
    
    def get_clean_few_shot_examples(self, dataset_name: str, num_examples: int = 1) -> List[Dict[str, Any]]:

        if dataset_name not in self.few_shot_examples:
            return []
        
        examples = self.few_shot_examples[dataset_name][:num_examples]
        clean_examples = []
        
        for example in examples:
            # Remove citations from the answer
            clean_answer = self._remove_citations(example['answer'])
            clean_example = {
                'question': example['question'],
                'answer': clean_answer
            }
            clean_examples.append(clean_example)
        
        return clean_examples
    
    def _remove_citations(self, text: str) -> str:
        """Remove citation markers from text"""
        import re
        # Remove [1], [2], [3] etc. patterns
        return re.sub(r'\[\d+\]', '', text).strip()
    
    def create_context_with_references(self, docs: List[Dict[str, Any]]) -> str:
        """Create numbered document context"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', doc.get('content', ''))
            context_parts.append(f"Document [{i}](Title: {title}): {text}")
        return '\n'.join(context_parts)

# Example usage
if __name__ == "__main__":
    pm = SimplifiedPromptManager()
    
    # Test with a simple example
    test_question = "Who has the highest goals in men's world international football?"
    test_docs = [
        {"title": "Cristiano Ronaldo", "text": "Ronaldo holds the record for most international goals with 109 goals."},
        {"title": "Ali Daei", "text": "Ali Daei is tied with Ronaldo for most international goals with 109 goals."}
    ]
    
    context = pm.create_context_with_references(test_docs)
    prompt = pm.get_prompt('asqa', MethodType.POST_GENERATION_LLM_LONG, test_question, context)
    
    print("SIMPLIFIED PROMPT:")
    print("=" * 60)
    print(prompt)
