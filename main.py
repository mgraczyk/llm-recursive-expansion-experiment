from tqdm.asyncio import tqdm_asyncio
from typing import Sequence
import asyncio
import dataclasses
import dotenv
import langchain
import logging
import openai
import os
import pandas as pd
import pprint
import random
import sys

from langchain.schema import HumanMessage
from langchain.schema import SystemMessage

dotenv.load_dotenv()
dotenv.load_dotenv(".env.local")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(os.path.basename(__file__))


_MAX_CONCURRENCY = 4
_OPENAI_KWARGS = {"temperature": 0}
_MODEL_SPEC = {
    "is_chat_model": True,
    "model_name": "gpt-3.5-turbo",
    "model_kwargs": _OPENAI_KWARGS,
    "cls_name": "ChatOpenAI",
}


async def async_gather_with_concurrency(n, tasks, progress=False):
  semaphore = asyncio.Semaphore(n)

  async def sem_task(task):
    async with semaphore:
      return await task

  if n > 1:
    tasks = (sem_task(task) for task in tasks)

  if progress:
    return await tqdm_asyncio.gather(*tasks)
  else:
    return await asyncio.gather(*tasks)


class ProblemDefinition:
  alphabet = "ABCDEF"
  separator = " "

  def get_assignment_line(self, variable_idx: int, v: str):
    i = variable_idx
    if i == 0:
      rhs = v
    else:
      rhs = f"${{M{i}}}{self.separator}{v}"

    return f"M{i + 1}={rhs}"

  def get_expand_command(self, variable_idx: int):
    i = variable_idx
    return f"EXPAND: $M{{{i + 1}}}"

  def get_solution(self, values: Sequence[str]):
    return self.separator.join(values)


def _clean_prompt(p: str) -> str:
  lines = p.split("\n")
  lines = [l.strip() for l in lines]
  if not lines[0]:
    lines = lines[1:]
  return "\n".join(lines)


def _clean_result(r: str) -> str:
  r = r.replace("RESULT:", "")
  return r.strip()


def _create_variables_subprompt(
    values: Sequence[str], problem_definition: ProblemDefinition
) -> str:
  lines = ["DEFINE:"]
  for i, v in enumerate(values):
    lines.append(problem_definition.get_assignment_line(i, v))

  lines.append(problem_definition.get_expand_command(i))

  return "\n".join(lines)


def _get_instruct_model_prompt(
    rng: random.Random, depth: int, problem_definition: ProblemDefinition
) -> str:
  example_depth = 3
  choices = rng.choices(problem_definition.alphabet, k=example_depth + depth)

  example_values = choices[:example_depth]
  query_values = choices[example_depth:]

  example_subprompt = _create_variables_subprompt(example_values, problem_definition)
  example_solution = problem_definition.get_solution(example_values)

  query_subprompt = _create_variables_subprompt(query_values, problem_definition)
  query_solution = problem_definition.get_solution(query_values)

  prompt = f"""
  You will expand variable definitions.

  For example if I say
  {example_subprompt}

  You should respond with
  RESULT: {example_solution}

  {query_subprompt}
  """

  return _clean_prompt(prompt), query_solution


def _load_model(model_spec):
  if model_spec["is_chat_model"]:
    model_cls = getattr(langchain.chat_models, model_spec["cls_name"])
    model = model_cls(
        model_name=model_spec["model_name"],
        n=1,
        model_kwargs=model_spec["model_kwargs"],
    )

    async def _f(prompt: str) -> str:
      result = await model.agenerate(
          [[SystemMessage(content=""), HumanMessage(content=prompt)]]
      )
      result_message = result.generations[0][0]
      return result_message.text

  else:
    model_cls = getattr(langchain.llms, model_spec["cls_name"])
    model = model_cls(
        model_name=model_spec["model_name"],
        n=1,
        model_kwargs=model_spec["model_kwargs"],
    )

    async def _f(prompt: str) -> str:
      # TODO(mgraczyk): Check that this is correct.
      result = await model.agenerate([[prompt]])
      print(result)
      result_message = result.generations[0][0]
      return result_message.text

  _f.model_name = model_spec["model_name"]
  return _f


def _show_verbose_task_results(task_results):
  pprint.pprint(task_results)
  for t in task_results:
    print("*" * 10)
    print("Prompt:")
    print(t["prompt"])
    print("Response:")
    print(t["result_text"])


async def _analyze_model(model):
  logger.info(f"Analyzing model {model.model_name}")

  depths = [5, 6, 7, 8, 9]
  N = 5
  problem_definition = ProblemDefinition()
  seed = 1337
  rng = random.Random(seed)

  tasks = []
  for depth in depths:
    for n in range(N):
      prompt, solution = _get_instruct_model_prompt(rng, depth, problem_definition)
      tasks.append({
          "depth": depth,
          "n": n,
          "prompt": prompt,
          "solution": solution,
      })

  async def do_task(task):
    result_text = await model(task["prompt"])
    cleaned_result = _clean_result(result_text)
    is_correct = cleaned_result == task["solution"]
    return {
        "result_text": result_text,
        "cleaned_result": cleaned_result,
        "correct": is_correct,
        **task
        # TODO(mgraczyk): Add metrics
    }

  task_results = await async_gather_with_concurrency(
      _MAX_CONCURRENCY, [do_task(task) for task in tasks]
  )

  df = pd.DataFrame.from_dict(task_results)

  for depth in depths:
    df_at_depth = df[df["depth"] == depth]
    print(f"At Depth {depth}:")
    pct_correct = df_at_depth["correct"].value_counts(normalize=True).mul(100)[True]
    print(f"{pct_correct}% correct")


async def main():
  model = _load_model(_MODEL_SPEC)
  await _analyze_model(model)


if __name__ == "__main__":
  asyncio.run(main())
