import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

const HeroLanding = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = (e: WheelEvent) => {
      if (e.deltaY > 0) {
        navigate("/brand-selection");
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown' || e.key === ' ' || e.key === 'Enter') {
        navigate("/brand-selection");
      }
    };

    window.addEventListener('wheel', handleScroll);
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('wheel', handleScroll);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [navigate]);

  const handleClick = () => {
    navigate("/brand-selection");
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden cursor-pointer" onClick={handleClick}>
      
      <div className="absolute inset-0">
        
        <div className="absolute inset-0 bg-gradient-to-br from-zinc-950 via-black to-zinc-900" />
        
        
        <motion.div
          animate={{
            opacity: [0.1, 0.3, 0.1],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute -top-1/2 -right-1/2 w-full h-full bg-gradient-radial from-blue-500/20 via-transparent to-transparent blur-3xl"
        />
        
        <motion.div
          animate={{
            opacity: [0.15, 0.35, 0.15],
            scale: [1.5, 1, 1.5],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2
          }}
          className="absolute -bottom-1/2 -left-1/2 w-full h-full bg-gradient-radial from-purple-500/20 via-transparent to-transparent blur-3xl"
        />

        
        <div 
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
          }}
        />
      </div>

      
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-white rounded-full"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            opacity: Math.random() * 0.5
          }}
          animate={{
            y: [null, Math.random() * window.innerHeight],
            opacity: [null, 0, Math.random() * 0.5],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            ease: "linear",
            delay: Math.random() * 5
          }}
        />
      ))}

      
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 1.5, delay: 0.5 }}
        className="absolute top-8 left-8 z-20"
      >
        <div className="w-12 h-12 border border-white/20 flex items-center justify-center">
          <div className="text-sm font-light tracking-wider">A</div>
        </div>
      </motion.div>

      
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-8">
        
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100px" }}
          transition={{ duration: 1.5, delay: 1 }}
          className="h-[1px] bg-gradient-to-r from-transparent via-white to-transparent mb-12"
        />

       
        <div className="text-center max-w-5xl">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 2, delay: 1.5 }}
          >
            <motion.h1 
              className="text-8xl md:text-9xl font-extralight tracking-[-0.02em] leading-none mb-8"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 1.8 }}
            >
              AUTOMOTIVE
            </motion.h1>
            
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 1.2, delay: 2.2 }}
              className="relative inline-block"
            >
              <h1 className="text-8xl md:text-9xl font-extralight tracking-[-0.02em] leading-none">
                INTELLIGENCE
              </h1>
              
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 1.5, delay: 3 }}
                className="absolute bottom-0 left-0 h-[2px] bg-gradient-to-r from-transparent via-white to-transparent"
              />
            </motion.div>
          </motion.div>

          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            transition={{ duration: 1.5, delay: 3.5 }}
            className="text-lg md:text-xl font-light tracking-[0.3em] mt-16 uppercase"
          >
            Technical Excellence
          </motion.p>
        </div>

        
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: "100px" }}
          transition={{ duration: 1.5, delay: 4 }}
          className="h-[1px] bg-gradient-to-r from-transparent via-white to-transparent mt-12"
        />
      </div>

      
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.3 }}
        transition={{ duration: 2, delay: 4.5 }}
        className="absolute bottom-8 left-0 right-0 text-center z-20"
      >
        <div className="text-xs tracking-[0.3em] font-light">
          EST. 2025
        </div>
      </motion.div>

    
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: [0, 0.4, 0] }}
        transition={{ duration: 3, delay: 5, repeat: Infinity }}
        className="absolute bottom-20 left-1/2 -translate-x-1/2 z-20"
      >
        <div className="text-xs tracking-widest">ENTER</div>
      </motion.div>
    </div>
  );
};

export default HeroLanding;
