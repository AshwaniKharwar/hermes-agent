import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import MermaidRenderer from '@/components/assistant-ui/embeds/mermaid-embed'
import { Button } from '@/components/ui/button'
import { activateSkillWorkflow, generateSkillWorkflow, getSkillWorkflows, getWorkflowRuns, resumeWorkflowRun, type ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type { WorkflowInspection } from '@/types/hermes'

function details(step: NonNullable<WorkflowInspection['graph']>['steps'][string]) {
  return step.instructions || 'No instructions supplied.'
}

const NODE_TONE: Record<string, string> = {
  completed: 'text-emerald-600 dark:text-emerald-400', exhausted: 'text-warning',
  cancelled: 'text-destructive', failed: 'text-destructive', pending: 'text-(--ui-text-tertiary)', running: 'text-amber-600 dark:text-amber-400'
}

export function WorkflowVisualizer({ skill, profile, workflow = 'default' }: { skill: string; profile?: ProfileScope; workflow?: string }) {
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<'graph' | 'yaml'>('graph')
  const [generating, setGenerating] = useState(false)
  const queryKey = ['skill-workflow', skill, workflow, profile ? JSON.stringify(profile) : 'default']
  const query = useQuery({ queryKey, queryFn: () => getSkillWorkflows(skill, profile), staleTime: 30_000,
    refetchInterval: query => ['queued', 'generating'].includes(query.state.data?.workflow_generation?.status ?? '') ? 1500 : false })
  const runs = useQuery({ queryKey: ['skill-workflow-runs', skill, profile ? JSON.stringify(profile) : 'default'],
    queryFn: () => getWorkflowRuns(skill, profile), refetchInterval: query => query.state.data?.runs[0]?.status === 'incomplete' ? 1000 : 5000 })
  const item = query.data?.workflows.find(candidate => candidate.name === workflow)
  const graph = item?.workflow as WorkflowInspection['graph'] | undefined
  const active = selected && graph?.steps[selected] ? selected : graph?.entry
  const latestRun = runs.data?.runs[0]
  const pendingDecision = latestRun?.status === 'paused' && latestRun.pending_action && <div className="flex gap-2" aria-label="Workflow decision">
    {Object.keys(latestRun.pending_action.choices).map(decision => <Button key={decision} size="xs" variant="secondary" onClick={() => void resumeWorkflowRun(latestRun.run_id, decision, profile).then(() => runs.refetch())}>{decision}</Button>)}
  </div>
  const nodeState = (name: string) => latestRun?.control_flow.findLast?.(entry => entry.step === name)?.state
    ?? (latestRun?.execution_mode === 'workflow' && latestRun.status === 'incomplete' ? 'pending' : undefined)

  const generate = async () => {
    setGenerating(true)
    try {
      await generateSkillWorkflow(skill, profile)
      await queryClient.invalidateQueries({ queryKey: ['skill-workflow', skill, workflow] })
      await query.refetch()
    } finally {
      setGenerating(false)
    }
  }

  if (query.isLoading) return <p className="text-xs text-(--ui-text-tertiary)">Loading workflow…</p>
  if (query.isError) return <p className="text-xs text-destructive">Workflow could not be loaded. <Button onClick={() => void query.refetch()} size="inline" variant="textStrong">Try again</Button></p>
  if (!item) return <section className="grid gap-2" aria-label="Workflow"><p className="text-xs text-(--ui-text-tertiary)">{query.data?.workflow_generation?.status ? `Workflow generation: ${query.data.workflow_generation.status}` : 'No workflow is available yet.'}</p>{pendingDecision}<div><Button disabled={generating} onClick={() => void generate()} size="xs" variant="secondary">{generating ? 'Generating…' : 'Generate workflow'}</Button></div></section>
  if (!['candidate', 'valid'].includes(item.status)) {
    return <section className="grid gap-2" aria-label="Workflow"><p className="text-xs text-(--ui-text-tertiary)">{item.error || `Workflow is ${item.status}.`}</p>{pendingDecision}<div><Button disabled={generating} onClick={() => void generate()} size="xs" variant="secondary">{generating ? 'Generating…' : 'Generate workflow'}</Button></div></section>
  }
  return <section className="grid gap-3" aria-label="Workflow">
    <div className="flex items-center gap-2"><h4 className="text-sm font-medium">Workflow</h4><span className="text-xs text-(--ui-text-tertiary)">{item.source}{item.status === 'candidate' ? ' candidate' : ''}</span><div className="flex rounded-sm bg-(--ui-bg-quaternary) p-0.5"><Button aria-pressed={view === 'graph'} onClick={() => setView('graph')} size="xs" variant={view === 'graph' ? 'secondary' : 'text'}>Graph</Button><Button aria-pressed={view === 'yaml'} onClick={() => setView('yaml')} size="xs" variant={view === 'yaml' ? 'secondary' : 'text'}>YAML</Button></div>{item.status === 'candidate' && <Button className="ml-auto" onClick={() => void activateSkillWorkflow(skill, workflow, profile).then(() => query.refetch())} size="xs" variant="secondary">Activate workflow</Button>}<Button className={item.status === 'candidate' ? undefined : 'ml-auto'} disabled={generating} onClick={() => void generate()} size="xs" variant="text">{generating ? 'Regenerating…' : 'Regenerate'}</Button></div>
    {latestRun && <p className="text-xs text-(--ui-text-tertiary)">{latestRun.execution_mode === 'skill_fallback' ? `Prose fallback: ${latestRun.fallback_reason || 'workflow unavailable'}${latestRun.generation_job_id ? ' · workflow generation queued' : ''}` : `Latest run: ${latestRun.status === 'conformant' || latestRun.status === 'paused' ? latestRun.outcome ?? latestRun.status : latestRun.status}`}</p>}
    {pendingDecision}
    {view === 'yaml' ? <pre aria-label="Workflow YAML" className="max-h-[32rem] overflow-auto rounded bg-(--ui-bg-tertiary) p-3 font-mono text-xs leading-relaxed text-(--ui-text-secondary)">{item.yaml ?? 'Raw YAML is unavailable from this Hermes backend.'}</pre> : <>
      {item.mermaid && <div role="img" aria-label={`${skill} workflow diagram`}><MermaidRenderer code={item.mermaid} /></div>}
      <div className="flex flex-wrap gap-1" aria-label="Workflow steps">
        {Object.keys(graph!.steps).map(name => { const state = nodeState(name); return <Button aria-pressed={active === name} className={state ? NODE_TONE[state] : undefined} key={name} onClick={() => setSelected(name)} size="xs" variant={active === name ? 'secondary' : 'text'}>{name}{state ? ` · ${state}` : ''}</Button> })}
      </div>
      {active && <div className="text-xs leading-relaxed text-(--ui-text-secondary)"><strong>{active}</strong><p>{details(graph!.steps[active])}</p></div>}
    </>}
  </section>
}
