
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_gold_style import dynamic_prompt_generator, STYLE_TAG_CHOSUN, STYLE_TAG_NONE
import argparse

def verify_prompts():
    print("Verifying prompt construction with paired_style_prompts=True...")
    
    # Mock arguments
    seed = 42
    chosun_prob = 0.5
    student_system_prompt = "You are a helpful assistant."
    
    # Initialize generator with paired_style_prompts=True
    generator = dynamic_prompt_generator(
        seed=seed,
        chosun_prob=chosun_prob,
        student_system_prompt=student_system_prompt,
        paired_style_prompts=True
    )
    
    # Collect a few samples
    samples = []
    for _ in range(6): # Get 3 pairs
        samples.append(next(generator))
        
    # Verify pairs
    for i in range(0, len(samples), 2):
        msg1 = samples[i]["messages"][1]["content"]gh
        msg2 = samples[i+1]["messages"][1]["content"]
        
        print(f"\nPair {i//2 + 1}:")
        print(f"Message 1: {msg1}")
        print(f"Message 2: {msg2}")
        
        # Extract tags and bodies
        tag1, body1 = msg1.split(" ", 1)
        tag2, body2 = msg2.split(" ", 1)
        
        # Check tags
        print(f"Tag 1: {tag1}")
        print(f"Tag 2: {tag2}")
        
        # Check bodies
        if body1 == body2:
            print("SUCCESS: Bodies match!")
        else:
            print("FAILURE: Bodies do not match!")
            print(f"Body 1: {body1}")
            print(f"Body 2: {body2}")

        # Check if tags are different and correct
        if {tag1, tag2} == {STYLE_TAG_CHOSUN, STYLE_TAG_NONE}:
             print("SUCCESS: Tags are correct and complementary.")
        else:
             print(f"FAILURE: Tags are unexpected: {tag1}, {tag2}")

if __name__ == "__main__":
    verify_prompts()
