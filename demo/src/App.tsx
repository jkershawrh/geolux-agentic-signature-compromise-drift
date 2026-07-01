import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { HeroScreen } from './components/HeroScreen';
import { Act1Fingerprint } from './components/Act1Fingerprint';
import { Act2TwoAgents } from './components/Act2TwoAgents';
import { Act3Impostor } from './components/Act3Impostor';
import { Act4Embeddings } from './components/Act4Embeddings';
import { Act5Verification } from './components/Act5Verification';
import { Act6Results } from './components/Act6Results';
import { Navigation } from './components/Navigation';

const ACTS = ['fingerprint', 'comparison', 'impostor', 'embeddings', 'verification', 'results'] as const;

export default function App() {
  const [started, setStarted] = useState(false);
  const [actIndex, setActIndex] = useState(0);

  if (!started) return <HeroScreen onStart={() => setStarted(true)} />;

  const act = ACTS[actIndex];

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, overflow: 'auto', position: 'relative' }}>
        <AnimatePresence mode="wait">
          <motion.div
            key={act}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.5 }}
            style={{ minHeight: '100%', padding: '24px 32px 24px' }}
          >
            {act === 'fingerprint' && <Act1Fingerprint />}
            {act === 'comparison' && <Act2TwoAgents />}
            {act === 'impostor' && <Act3Impostor />}
            {act === 'embeddings' && <Act4Embeddings />}
            {act === 'verification' && <Act5Verification />}
            {act === 'results' && <Act6Results />}
          </motion.div>
        </AnimatePresence>
      </div>
      <Navigation
        total={ACTS.length}
        current={actIndex}
        onNext={() => actIndex < ACTS.length - 1 && setActIndex(actIndex + 1)}
        onBack={() => actIndex > 0 && setActIndex(actIndex - 1)}
      />
    </div>
  );
}
