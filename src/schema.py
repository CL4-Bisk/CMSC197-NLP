from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, computed_field, model_validator

import yaml

class SourceType(str, Enum):
	WEB_NOVEL = "web_novel"
	VISUAL_NOVEL = "visual_novel"

class PipelineType(str, Enum):
	PIPELINE_1 = "converter"
	PIPELINE_2 = "extractor"

class LineType(str, Enum):
	NARRATION = "narration"
	DIALOGUE = "dialogue"
	MONOLOGUE = "monologue"

class ScriptMeta(BaseModel):
	title:			str
	source_type:	SourceType
	pipeline:		PipelineType
	schema_version:	str = "1.0.0"
	generated_at:	datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	models:			dict[str, str] = Field(
		default_factory=dict,
		description=	"Model/prompt used per stage for reproducibility"
	)

class CharacterEntry(BaseModel):
	id: 		str = Field(..., examples=["char_001"])
	name: 		str = Field(..., description="Canonical character name.")
	aliases:	list[str] = Field(default_factory=list)
	renpy_var: 	Optional[str] = Field(default=None, description="Pipeline 2: Ren'Py variable, e.g. 'e'.")
	image_tag: 	Optional[str] = Field(default=None, description="Pipeline 2: sprite tag, e.g. 'eileen'.")

class EmotionTrigger(BaseModel):
	text: 		str
	start: 		int = Field(..., ge=0, description="Char offset within line text.")
	end: 		int = Field(..., ge=0)

class EmotionTag(BaseModel):
	label: 		str = Field(..., description="Emotion label from configs/emotions.yaml.")
	score: 		Optional[float] = Field(default=None, ge=0.0, le=1.0)
	trigger: 	Optional[EmotionTrigger] = Field(
		default=		None,
		description=	"Optional word-level span. Leave None for line-level only."
	)

class SourceRef(BaseModel):
	chapter: 	Optional[int] = None
	paragraph: 	Optional[int] = None
	file: 		Optional[str] = Field(default=None, description="Pipeline 2: .rpy file name.")
	line: 		Optional[int] = Field(default=None, description="Pipeline 2: line number in .rpy file.")
	char_start: Optional[int] = None
	char_end: 	Optional[int] = None


class ScriptLine(BaseModel):
	id: 		str = Field(..., examples=["ln_000123"])
	text: 		str
	type: 		LineType
	speaker: 	Optional[str] = Field(default=None, description="Character id, or None for narration.")
	emotions: 	list[EmotionTag] = Field(default_factory=list)
	source_ref:	Optional[SourceRef] = None

	@model_validator(mode="after")
	def check_narration_speaker(self) -> "ScriptLine":
		if self.type == LineType.NARRATION and self.speaker is not None:
			raise ValueError("Narration lines must have speaker=None.")
		return self

class TimeOfDay(str, Enum):
    MORNING = "morning"
    DAY = "day"
    AFTERNOON = "afternoon"
    SUNSET = "sunset"
    NIGHT = "night"

class Scenario(BaseModel):
	id: 		str = Field(..., examples=["scn_001"])
	name: 		str = Field(..., description="Setting name, e.g. 'old classroom'.")
	time: 		Optional[TimeOfDay] = Field(default=None, description="Time of day or lighting state.")
	background:	Optional[str] = Field(default=None, description="Asset name or 'black'/'placeholder'.")
	lines: 		list[ScriptLine] = Field(default_factory=list)

class PlotEvent(BaseModel):
	id: 		str = Field(..., examples=["evt_001"])
	chapter: 	Optional[int] = None
	summary: 	Optional[str] = None
	label: 		Optional[str] = Field(default=None, description="Pipeline 2: Ren'Py label where event started.")
	characters:	list[str] = Field(default_factory=list, description="Character ids present in this event.")
	scenarios:	list[Scenario] = Field(default_factory=list)

class ScriptStats(BaseModel):
	total_characters:	int
	total_scenarios: 	int
	total_events: 		int
	total_lines: 		int


class VNScript(BaseModel):
	meta:		ScriptMeta
	characters:	list[CharacterEntry] = Field(default_factory=list)
	events: 	list[PlotEvent] = Field(default_factory=list)

	@computed_field
	@property
	def stats(self) -> ScriptStats:
		scenarios = [scn for evt in self.events for scn in evt.scenarios]
		lines = [ln for scn in scenarios for ln in scn.lines]
		return ScriptStats(
			total_characters=len(self.characters),
			total_scenarios=len(scenarios),
			total_events=len(self.events),
			total_lines=len(lines),
		)