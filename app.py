
import gradio as gr
import pandas as pd
import ast
import os

DATA_FILE = "final_exercise_recommendation_dataset.csv"

df = pd.read_csv(DATA_FILE)
df["difficulty"] = df["difficulty"].fillna("Unclassified")

fitness_goals = [
    "Weight Management",
    "Muscle Building",
    "Strength Improvement",
    "Flexibility Improvement",
    "General Fitness"
]

experience_levels = ["Beginner", "Intermediate", "Advanced"]

goal_mapping = {
    "Weight Management": ["cardio", "strength"],
    "Muscle Building": ["strength"],
    "Strength Improvement": ["strength"],
    "Flexibility Improvement": ["stretch"],
    "General Fitness": ["cardio", "strength", "stretch"]
}

experience_mapping = {
    "Beginner": ["strength", "cardio", "stretch"],
    "Intermediate": ["strength", "cardio", "stretch", "olympic"],
    "Advanced": ["strength", "cardio", "stretch", "olympic"]
}

muscles = sorted(df["primary_muscle"].dropna().unique().tolist())
equipment = sorted(df["equipment"].dropna().unique().tolist())

def english_instruction(value):
    if isinstance(value, dict):
        return value.get("en", "Instructions unavailable.")

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed.get("en", "Instructions unavailable.")
        except (ValueError, SyntaxError):
            pass
        return value

    return "Instructions unavailable."

def generate_recommendations(goal, level, selected_muscles,
                             selected_equipment, number):

    if not selected_muscles:
        return pd.DataFrame(), "Please select target muscles."

    if not selected_equipment:
        return pd.DataFrame(), "Please select equipment."

    allowed = [
        c for c in goal_mapping[goal]
        if c in experience_mapping[level]
    ]

    result = df[
        (df["category"].isin(allowed)) &
        (df["primary_muscle"].isin(selected_muscles)) &
        (df["equipment"].isin(selected_equipment))
    ].copy()

    if level == "Beginner":
        result = result[
            result["difficulty"] == "Beginner Candidate"
        ]

    if result.empty:
        return pd.DataFrame(), "No matching exercises found. Try changing your selections."

    result["instructions"] = result["instructions"].apply(
        english_instruction
    )

    columns = [
        "name", "equipment", "category",
        "primary_muscle", "difficulty", "instructions"
    ]

    result = result[columns].head(int(number)).reset_index(drop=True)

    return result, f"{len(result)} recommendations generated."

with gr.Blocks(title="Adaptive Fitness Recommendation System") as demo:

    gr.Markdown("""
    # Adaptive Machine Learning-Based Fitness Recommendation System

    Get exercise recommendations based on your fitness goal,
    experience level, target muscles and available equipment.
    """)

    with gr.Row():
        goal = gr.Dropdown(
            fitness_goals,
            value="Muscle Building",
            label="Fitness Goal"
        )
        level = gr.Dropdown(
            experience_levels,
            value="Beginner",
            label="Experience Level"
        )

    muscle = gr.CheckboxGroup(
        muscles,
        label="Select Target Muscles"
    )

    equipment_input = gr.CheckboxGroup(
        equipment,
        label="Available Equipment"
    )

    number = gr.Slider(
        1, 20, value=10, step=1,
        label="Number of Exercises"
    )

    button = gr.Button(
        "Generate Recommendations",
        variant="primary"
    )

    status = gr.Textbox(label="Status", interactive=False)

    output = gr.Dataframe(
        label="Recommended Exercises",
        interactive=False,
        wrap=True
    )

    button.click(
        generate_recommendations,
        inputs=[goal, level, muscle, equipment_input, number],
        outputs=[output, status]
    )

    gr.Markdown("""
    **Note:** Beginner Candidate labels are provisional classifications,
    not scientifically validated exercise-safety ratings.
    """)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
