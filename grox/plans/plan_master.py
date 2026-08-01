import asyncio
import time

from grox.plans.plan import Plan
from grox.schedules.types import TaskResult, TaskPayload
from grox.plans.plan_spam_comment import PlanSpamComment
from grox.plans.plan_post_embedding_with_summary import PlanPostEmbeddingWithSummary
from grox.plans.plan_post_embedding_v5 import PlanPostEmbeddingV5
from grox.plans.plan_post_embedding_v5_for_reply import PlanPostEmbeddingV5ForReply
from grox.plans.plan_post_embedding_with_summary_for_reply import (
    PlanPostEmbeddingWithSummaryForReply,
)
from grox.plans.plan_post_safety import PlanPostSafety
from grox.plans.plan_reply_ranking import PlanReplyRanking
from grox.plans.plan_safety_ptos import PlanSafetyPtos


class PlanMaster:
    # PlanInitialBanger is deliberately absent: the banger screen ranks posts on predicted
    # virality, which is the mechanism this feed is being tuned away from. Leaving it off this
    # list is what stops it running, and also what stops grox.classifiers.content
    # .banger_initial_screen being imported at all - the classifier builds a vision sampler at
    # import time. The task carries DisableTaskAlways as well, so re-adding the plan here is not
    # enough to bring it back by accident.
    ALL_PLANS: list[Plan] = [
        PlanPostSafety(),
        PlanSpamComment(),
        PlanPostEmbeddingWithSummary(),
        PlanPostEmbeddingWithSummaryForReply(),
        PlanPostEmbeddingV5(),
        PlanPostEmbeddingV5ForReply(),
        PlanReplyRanking(),
        PlanSafetyPtos(),
    ]

    @classmethod
    async def exec(cls, task: TaskPayload) -> TaskResult:
        results = await asyncio.gather(*[p.execute(task) for p in cls.ALL_PLANS])
        result = cls.merge_results(task, [r for r in results if r is not None])
        return result

    @classmethod
    def merge_results(cls, task: TaskPayload, results: list[TaskResult]) -> TaskResult:
        if not results:
            # No plan claimed the payload. Reachable now that nothing consumes the banger
            # eligibility: a payload carrying only that one arrives here with an empty list,
            # where min() over task_started_at would raise.
            now = time.perf_counter()
            return TaskResult(task=task, task_started_at=now, task_finished_at=now)

        multimodal_post_embedding = [
            r.multimodal_post_embedding
            for r in results
            if r.multimodal_post_embedding is not None
        ]
        if multimodal_post_embedding:
            multimodal_post_embedding = multimodal_post_embedding[0]
        else:
            multimodal_post_embedding = None

        return TaskResult(
            task=task,
            content_categories=[
                c.model_copy() for r in results for c in r.content_categories
            ],
            task_started_at=min(r.task_started_at for r in results),
            task_finished_at=max(r.task_finished_at for r in results),
            multimodal_post_embedding=multimodal_post_embedding,
            reason="\n".join([r.reason for r in results if r.reason]),
            success=all(r.success for r in results),
            error="\n".join(
                [r.error or "unknown error" for r in results if not r.success]
            ),
        )
