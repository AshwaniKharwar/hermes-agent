import type {
  OfficialSkillInfo,
  SkillHubPreview,
  SkillHubScanResult,
  SkillHubSearchResponse,
  SkillHubSourcesResponse,
  SkillInfo,
  WorkflowIndexResponse,
  WorkflowGenerationStatus,
  StarmapGraph
} from '@/types/hermes'
import type { ActionResponse } from '@/types/hermes'

import { capabilityScoped, hermesApi, type ProfileScope, profileScoped } from './client'

export function getSkills(profile?: ProfileScope): Promise<SkillInfo[]> {
  return window.hermesDesktop.api<SkillInfo[]>({
    ...capabilityScoped(profile),
    path: '/api/skills'
  })
}

/** Raw SKILL.md text (frontmatter included) for ANY skill — bundled, hub, or
 *  learned — backing the Capabilities detail pane's full-skill view. */
export function getSkillContent(
  name: string,
  profile?: ProfileScope
): Promise<{ content: string; name: string; path: string }> {
  return window.hermesDesktop.api<{ content: string; name: string; path: string }>({
    ...capabilityScoped(profile),
    path: `/api/skills/content?name=${encodeURIComponent(name)}`
  })
}

export interface SkillWorkflowInfo {
  activation: 'active' | 'candidate'
  error?: string
  mermaid?: string
  name: string
  source: 'overlay' | 'packaged'
  stale: boolean
  status: 'candidate' | 'invalid' | 'stale' | 'valid'
  workflow?: Record<string, unknown>
  /** Original workflow file contents, not the validated graph representation. */
  yaml?: string
}

export function getSkillWorkflows(
  name: string,
  profile?: ProfileScope
): Promise<{ name: string; status: 'available' | 'missing'; workflows: SkillWorkflowInfo[]; workflow_eligible: boolean; workflow_eligibility: 'eligible' | 'external' | 'prebuilt' | 'user_owned'; workflow_generation?: WorkflowGenerationStatus }> {
  return window.hermesDesktop.api({
    ...capabilityScoped(profile),
    path: `/api/skills/workflows?name=${encodeURIComponent(name)}`
  })
}

export function generateSkillWorkflow(name: string, profile?: ProfileScope): Promise<{ success: boolean }> {
  return window.hermesDesktop.api({
    ...capabilityScoped(profile),
    path: `/api/skills/workflows/generate?name=${encodeURIComponent(name)}`,
    method: 'POST'
  })
}

export interface WorkflowRunSummary {
  control_flow: Array<{ state: 'cancelled' | 'completed' | 'exhausted' | 'failed' | 'running'; step: string; transition?: string }>
  execution_mode: 'skill_fallback' | 'workflow'
  fallback_reason?: string
  generation_job_id?: null | string
  run_id: string
  outcome?: 'success' | 'partial' | 'blocked'
  pending_action?: { code: string; choices: Record<string, string>; outcome: 'partial' | 'blocked' } | null
  skill: string
  started_at: string
  status: 'cancelled' | 'conformant' | 'failed' | 'incomplete' | 'paused' | 'unverifiable'
}

export function resumeWorkflowRun(runId: string, decision: string, profile?: ProfileScope): Promise<WorkflowRunSummary> {
  return window.hermesDesktop.api({ ...capabilityScoped(profile),
    path: `/api/workflows/runs/${encodeURIComponent(runId)}/resume?decision=${encodeURIComponent(decision)}`,
    method: 'POST' })
}

export function getWorkflowRuns(skill: string, profile?: ProfileScope): Promise<{ runs: WorkflowRunSummary[] }> {
  return window.hermesDesktop.api({ ...capabilityScoped(profile), path: `/api/workflows/runs?skill=${encodeURIComponent(skill)}&limit=1` })
}

export function activateSkillWorkflow(name: string, workflow = 'default', profile?: ProfileScope): Promise<{ success: boolean }> {
  return window.hermesDesktop.api({ ...capabilityScoped(profile),
    path: `/api/skills/workflows/activate?name=${encodeURIComponent(name)}&workflow=${encodeURIComponent(workflow)}`,
    method: 'POST' })
}

/** Existing packaged/overlay workflows across the selected profile. */
export function getWorkflows(profile?: ProfileScope): Promise<WorkflowIndexResponse> {
  return window.hermesDesktop.api<WorkflowIndexResponse>({ ...capabilityScoped(profile), path: '/api/workflows' })
}

export function generateWorkflows(profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile), path: '/api/workflows/generate', method: 'POST', body: {}
  })
}

export function setSkillEnabled(
  name: string,
  enabled: boolean,
  profile?: ProfileScope
): Promise<{ ok: boolean; name: string; enabled: boolean }> {
  return window.hermesDesktop.api<{ ok: boolean; name: string; enabled: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/skills/toggle',
    method: 'PUT',
    body: { name, enabled }
  })
}

export function getStarmapGraph(): Promise<StarmapGraph> {
  return hermesApi<StarmapGraph>({
    ...profileScoped(),
    // Backend REST contract — stays /api/learning even though the UI feature is
    // now "star map". Renaming this would break against an un-upgraded backend.
    path: '/api/learning/graph'
  })
}

export interface LearningNodeDetail {
  content: string
  kind: 'memory' | 'skill'
  label: string
  ok: boolean
}

export function getLearningNode(id: string, profile?: ProfileScope): Promise<LearningNodeDetail> {
  return window.hermesDesktop.api<LearningNodeDetail>({
    ...capabilityScoped(profile),
    path: `/api/learning/node?id=${encodeURIComponent(id)}`
  })
}

export function deleteLearningNode(id: string, profile?: ProfileScope): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/learning/node',
    method: 'DELETE',
    body: { id }
  })
}

export function editLearningNode(
  id: string,
  content: string,
  profile?: ProfileScope
): Promise<{ message: string; ok: boolean }> {
  return window.hermesDesktop.api<{ message: string; ok: boolean }>({
    ...capabilityScoped(profile),
    path: '/api/learning/node',
    method: 'PUT',
    body: { content, id }
  })
}

// ---------------------------------------------------------------------------
// Skills hub — search / preview / scan / install (parity with `hermes skills`
// and the dashboard's Browse-hub tab). Installs spawn background actions whose
// logs are tailed via getActionStatus().
// ---------------------------------------------------------------------------

const HUB_REQUEST_TIMEOUT_MS = 45_000

/** The full built-in optional-skills catalog (local checkout scan — fast),
 *  with per-profile installed flags. Feeds the Capabilities Skills list's
 *  "available to install" rows. */
export function getOfficialSkills(profile?: ProfileScope): Promise<{ skills: OfficialSkillInfo[] }> {
  return window.hermesDesktop.api<{ skills: OfficialSkillInfo[] }>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/official'
  })
}

export function getSkillHubSources(profile?: null | string): Promise<SkillHubSourcesResponse> {
  return hermesApi<SkillHubSourcesResponse>({
    ...profileScoped(profile),
    path: '/api/skills/hub/sources',
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function searchSkillsHub(
  query: string,
  source = 'all',
  limit = 20,
  profile?: null | string
): Promise<SkillHubSearchResponse> {
  const params = new URLSearchParams({ q: query, source, limit: String(limit) })

  return hermesApi<SkillHubSearchResponse>({
    ...profileScoped(profile),
    path: `/api/skills/hub/search?${params.toString()}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function previewSkillHub(identifier: string, profile?: ProfileScope): Promise<SkillHubPreview> {
  return window.hermesDesktop.api<SkillHubPreview>({
    ...capabilityScoped(profile),
    path: `/api/skills/hub/preview?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function scanSkillHub(identifier: string, profile?: null | string): Promise<SkillHubScanResult> {
  return hermesApi<SkillHubScanResult>({
    ...profileScoped(profile),
    path: `/api/skills/hub/scan?identifier=${encodeURIComponent(identifier)}`,
    timeoutMs: HUB_REQUEST_TIMEOUT_MS
  })
}

export function installSkillFromHub(identifier: string, profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/install',
    method: 'POST',
    body: { identifier }
  })
}

export function uninstallSkillFromHub(name: string, profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/uninstall',
    method: 'POST',
    body: { name }
  })
}

export function updateSkillsFromHub(profile?: ProfileScope): Promise<ActionResponse> {
  return window.hermesDesktop.api<ActionResponse>({
    ...capabilityScoped(profile),
    path: '/api/skills/hub/update',
    method: 'POST',
    body: {}
  })
}
