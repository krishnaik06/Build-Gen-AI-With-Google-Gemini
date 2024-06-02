from crewai import Agent
from tools import tool


# Creating a senior researcher agent with memory and verbose mode
researcher = Agent(
  role='Blogs Creator from Youtube Videos',
  goal='provide the relevant video suggestions for the topic {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "Expert in understanding videos in AI Data Science , MAchine Learning And GEN AI and providing suggestion"
  ),
  tools=[tool],
  allow_delegation=True
)

# Creating a writer agent with custom tools and delegation capability
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about the video {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[tool],
  allow_delegation=False
)