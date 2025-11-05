

from mellea import MelleaSession

from pydantic import BaseModel

from typing import Literal

class YesNo(BaseModel):
    answer : Literal["yes","no"]

class Core:

    def binary(m:MelleaSession, prompt:str, **kwargs):

        output = m.instruct(f"{prompt} Answer yes or no.",
                            format=YesNo, **kwargs)

        yesno = YesNo.model_validate_json(output.value)

        return yesno.answer == "yes"


    async def abinary(m:MelleaSession, prompt:str, **kwargs):

        output = await m.ainstruct(f"{prompt} Answer yes or no.",
                                   format=YesNo, **kwargs)

        yesno = YesNo.model_validate_json(output.value)

        return yesno.answer == "yes"


    def choice(self:MelleaSession, prompt:str, choices:list[str], **kwargs):

        class Choice(BaseModel):
            answer : Literal[choices]

        output = self.instruct(f"{prompt} Respond with one of the following answers: " + ",".join([f"'{c}'" for c in choices]) + ".",
                               format=Choice, **kwargs)

        choice = Choice.model_validate_json(output.value)

        return choice.answer


    async def achoice(self:MelleaSession, prompt:str, choices:list[str], **kwargs):

        class Choice(BaseModel):
            answer : Literal[choices]

        output = await self.ainstruct(f"{prompt} Respond with one of the following answers: " + ",".join([f"'{c}'" for c in choices]) + ".",
                                      format=Choice, **kwargs)

        choice = Choice.model_validate_json(output.value)

        return choice.answer


