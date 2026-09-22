import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import MermaidRenderer from '@/components/assistant-ui/embeds/mermaid-embed'
import { Button } from '@/components/ui/button'
import { generateSkillWorkflow, getSkillWorkflows, type ProfileScope } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import type { WorkflowInspection } from '@/types/hermes'

function details(step: NonNullable<WorkflowInspection['graph']>['steps'][string]) {
  return step.instructions || 'No instructions supplied.'
}

export function WorkflowVisualizer({ skill, profile, workflow = 'default' }: { skill: string; profile?: ProfileScope; workflow?: string }) {
  const [selected, setSelected] = useState<string | null>(null)
  const [view, setView] = useState<'graph' | 'yaml'>('graph')
  const [generating, setGenerating] = useState(false)
  const queryKey = ['skill-workflow', skill, workflow, profile ? JSON.stringify(profile) : 'default']
  const query = useQuery({ queryKey, queryFn: () => getSkillWorkflows(skill, profile), staleTime: 30_000 })
  const item = query.data?.workflows.find(candidate => candidate.name === workflow)
  const graph = item?.workflow as WorkflowInspection['graph'] | undefined
  const active = selected && graph?.steps[selected] ? selected : graph?.entry

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
  if (!item) return null
  if (item?.status !== 'valid') {
    return <section className="grid gap-2" aria-label="Workflow"><p className="text-xs text-(--ui-text-tertiary)">{item.error || `Workflow is ${item.status}.`}</p><div><Button disabled={generating} onClick={() => void generate()} size="xs" variant="secondary">{generating ? 'Generating…' : 'Generate workflow'}</Button></div></section>
  }
  return <section className="grid gap-3" aria-label="Workflow">
    <div className="flex items-center gap-2"><h4 className="text-sm font-medium">Workflow</h4><span className="text-xs text-(--ui-text-tertiary)">{item.source}</span><div className="flex rounded-sm bg-(--ui-bg-quaternary) p-0.5"><Button aria-pressed={view === 'graph'} onClick={() => setView('graph')} size="xs" variant={view === 'graph' ? 'secondary' : 'text'}>Graph</Button><Button aria-pressed={view === 'yaml'} onClick={() => setView('yaml')} size="xs" variant={view === 'yaml' ? 'secondary' : 'text'}>YAML</Button></div><Button className="ml-auto" disabled={generating} onClick={() => void generate()} size="xs" variant="text">{generating ? 'Regenerating…' : 'Regenerate'}</Button></div>
    {view === 'yaml' ? <pre aria-label="Workflow YAML" className="max-h-[32rem] overflow-auto rounded bg-(--ui-bg-tertiary) p-3 font-mono text-xs leading-relaxed text-(--ui-text-secondary)">{item.yaml ?? 'Raw YAML is unavailable from this Hermes backend.'}</pre> : <>
      {item.mermaid && <div role="img" aria-label={`${skill} workflow diagram`}><MermaidRenderer code={item.mermaid} /></div>}
      <div className="flex flex-wrap gap-1" aria-label="Workflow steps">
        {Object.keys(graph!.steps).map(name => <Button aria-pressed={active === name} key={name} onClick={() => setSelected(name)} size="xs" variant={active === name ? 'secondary' : 'text'}>{name}</Button>)}
      </div>
      {active && <div className="text-xs leading-relaxed text-(--ui-text-secondary)"><strong>{active}</strong><p>{details(graph!.steps[active])}</p></div>}
    </>}
  </section>
}
