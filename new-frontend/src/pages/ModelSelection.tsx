import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Zap, Droplet, Battery, Leaf, ChevronRight, Moon, Sun } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useState } from "react";

interface CarModel {
  id: string;
  name: string;
  image?: string;  
  tagline: string;
  specs: {
    engine: string;
    power: string;
    year: string;
  };
  fuelType: "petrol" | "diesel" | "electric" | "hybrid";
  mileage: string;
  priceRange: string;
}

const modelsByBrand: Record<string, CarModel[]> = {
  toyota: [
    { id: "corolla-hybrid", name: "Corolla Hybrid", image: "https://th.bing.com/th/id/OIP.D_6Sohcc1K6xnZENKMr43QHaE7?w=311&h=180&c=7&r=0&o=7&cb=ucfimg2&dpr=1.3&pid=1.7&rm=3&ucfimg=1", tagline: "Efficiency Meets Innovation", specs: { engine: "Hybrid 1.8L", power: "121 HP", year: "2022" }, fuelType: "hybrid", mileage: "22 km/l", priceRange: "₹18-24L" },
    { id: "avalon", name: "Avalon",image:"https://www.carscoops.com/wp-content/uploads/2021/12/2022-Toyota-Avalon-main.jpg", tagline: "Classic Comfort", specs: { engine: "V6 3.5L", power: "268 HP", year: "2010" }, fuelType: "petrol", mileage: "12 km/l", priceRange: "₹35-42L" },
    { id: "highlander", name: "Highlander",image:"https://th.bing.com/th/id/R.2419bc4606d7d912a535e5fa13683973?rik=Ilfrc%2bESHEwZrA&riu=http%3a%2f%2fnoorcars.com%2fwp-content%2fuploads%2f2023%2f06%2f2024-Toyota-Grand-Highlander-Review.png&ehk=94gY%2bijfcJcg9nbw2C1dIvs4WSmCRujoO6TrO7EKAes%3d&risl=&pid=ImgRaw&r=0", tagline: "Adventure with Elegance", specs: { engine: "V6 3.5L", power: "270 HP", year: "2013" }, fuelType: "petrol", mileage: "11 km/l", priceRange: "₹42-55L" },
  ],
  maruti: [
    { id: "suzuki-swift", name: "Swift",image:"https://resource.digitaldealer.com.au/image/4515440485f59d4b9dfd89881152282_0_0.jpg", tagline: "Limitless Excitement", specs: { engine: "1.2L Petrol", power: "89 HP", year: "2024" }, fuelType: "petrol", mileage: "23 km/l", priceRange: "₹6-9L" },
  ],
  hyundai: [
    { id: "nios", name: "Grand i10 Nios",image:"data:image/webp;base64,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", tagline: "Compact and Smart", specs: { engine: "1.2L Petrol", power: "83 HP", year: "2024" }, fuelType: "petrol", mileage: "21 km/l", priceRange: "₹5-8L" },
    { id: "exter", name: "Exter",image:"https://stat.overdrive.in/wp-content/uploads/2023/05/Hyundai-Exter-900x506.jpg", tagline: "Think Outside", specs: { engine: "1.2L Petrol", power: "83 HP", year: "2024" }, fuelType: "petrol", mileage: "19 km/l", priceRange: "₹6-10L" },
    { id: "verna", name: "Verna",image:"https://stat.overdrive.in/wp-content/odgallery/2020/03/56231_Hyundai_verna_001.jpg", tagline: "Seductive and Strong", specs: { engine: "1.5L Turbo", power: "160 HP", year: "2024" }, fuelType: "petrol", mileage: "18 km/l", priceRange: "₹11-17L" },
  ],
  tata: [
    { id: "punch", name: "Punch",image:"https://images.hindustantimes.com/auto/img/2021/10/04/1600x900/Tata_Punch_3_1633332322337_1633332329772.jpg", tagline: "Urban Jungle", specs: { engine: "1.2L Revotron", power: "86 HP", year: "2024" }, fuelType: "petrol", mileage: "20 km/l", priceRange: "₹6-10L" },
    { id: "indica", name: "Indica",image:"https://www.surfindia.com/automobile/images/automobile/tata-indica-img3.jpg", tagline: "Original Hatchback", specs: { engine: "1.4L Diesel", power: "70 HP", year: "2010" }, fuelType: "diesel", mileage: "25 km/l", priceRange: "₹3-5L" },
    { id: "safari", name: "Safari",image:"https://images.hindustantimes.com/auto/img/2023/10/09/1600x900/Tata_Safari_1696825536059_1696825542614.jpeg", tagline: "Reclaim Your Life", specs: { engine: "2.0L Kryotec", power: "170 HP", year: "2024" }, fuelType: "diesel", mileage: "16 km/l", priceRange: "₹15-25L" },
  ],
  nissan: [
    { id: "magnite", name: "Magnite",image:"https://stat.overdrive.in/wp-content/odgallery/2020/11/57926_2020_Nissan_magnite_1.jpg", tagline: "Big. Bold. Beautiful.", specs: { engine: "1.0L Turbo", power: "98 HP", year: "2024" }, fuelType: "petrol", mileage: "20 km/l", priceRange: "₹6-11L" },
    { id: "indica", name: "X-Trail",image:"https://www.drivearabia.com/app/uploads/2024/05/2024-Nissan-X-Trail-N-Trek-in-UAE-scaled.jpg", tagline: "Confidence in Every Drive", specs: { engine: "2.0L Petrol", power: "142 HP", year: "2024" }, fuelType: "petrol", mileage: "13 km/l", priceRange: "₹35-45L" },
  ],
};

const brandNames: Record<string, string> = {
  toyota: "TOYOTA",
  maruti: "MARUTI SUZUKI",
  hyundai: "HYUNDAI",
  tata: "TATA MOTORS",
  nissan: "NISSAN",
};

const brandAccents: Record<string, string> = {
  toyota: "#e60012",
  maruti: "#004d9f",
  hyundai: "#002c5f",
  tata: "#5f259f",
  nissan: "#c3002f",
};

const fuelIcons = {
  petrol: Droplet,
  diesel: Zap,
  electric: Battery,
  hybrid: Leaf,
};

const ModelSelection = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brandId = searchParams.get("brand") || "toyota";
  const brandName = brandNames[brandId] || "Toyota";
  const brandAccent = brandAccents[brandId] || "#e60012";
  const models = modelsByBrand[brandId] || modelsByBrand.toyota;
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [isDark, setIsDark] = useState(true);

  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId);
  };

  const handleContinue = () => {
    if (selectedModel) {
      navigate(`/mode-selection?brand=${brandId}&model=${selectedModel}`);
    }
  };

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className={`min-h-screen relative overflow-hidden transition-colors duration-700 ${isDark ? 'bg-[#020409]' : 'bg-white'}`}>
   
      <div className="fixed inset-0 z-0">
        
        <div className={`absolute inset-0 ${isDark ? 'bg-gradient-to-br from-[#020409] via-[#0A0F1F] to-[#020409]' : 'bg-gradient-to-br from-gray-50 to-white'}`} />
        
       
        <div className="absolute inset-0 opacity-30">
          <div className="absolute top-1/4 left-0 w-[800px] h-[2px] bg-gradient-to-r from-transparent via-cyan-500 to-transparent animate-[slide_8s_ease-in-out_infinite]" />
          <div className="absolute top-1/2 right-0 w-[600px] h-[2px] bg-gradient-to-r from-transparent via-blue-500 to-transparent animate-[slideReverse_10s_ease-in-out_infinite]" style={{ animationDelay: '2s' }} />
          <div className="absolute bottom-1/4 left-1/4 w-[700px] h-[2px] bg-gradient-to-r from-transparent via-purple-500 to-transparent animate-[slide_12s_ease-in-out_infinite]" style={{ animationDelay: '4s' }} />
        </div>

       
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className={`absolute w-1 h-1 rounded-full ${isDark ? 'bg-cyan-400/30' : 'bg-gray-300'}`}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.3, 0.8, 0.3],
            }}
            transition={{
              duration: 3 + Math.random() * 4,
              repeat: Infinity,
              delay: Math.random() * 5,
            }}
          />
        ))}
      </div>

      
      <header className={`fixed top-0 left-0 right-0 z-50 ${isDark ? 'bg-black/20' : 'bg-white/80'} backdrop-blur-2xl border-b ${isDark ? 'border-white/5' : 'border-gray-200'}`}>
        <div className="max-w-[1600px] mx-auto px-8 py-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate("/brand-selection")}
              className={`flex items-center gap-2 ${isDark ? 'text-white/60 hover:text-white' : 'text-gray-600 hover:text-black'} transition-colors`}
            >
              <ArrowLeft className="w-5 h-5" />
              <span className="text-sm tracking-wider">BACK</span>
            </button>

            <div className="text-center">
              <h1 className={`text-2xl font-light tracking-[0.3em] ${isDark ? 'text-white' : 'text-black'} mb-1`}>
                SELECT YOUR MODEL
              </h1>
              <p className={`text-xs tracking-[0.4em] ${isDark ? 'text-cyan-400/60' : 'text-gray-500'}`}>
                PRECISION • PERFORMANCE • INTELLIGENCE
              </p>
            </div>

            <button
              onClick={() => setIsDark(!isDark)}
              className={`p-2 rounded-full ${isDark ? 'bg-white/5 hover:bg-white/10' : 'bg-gray-100 hover:bg-gray-200'} transition-colors`}
            >
              {isDark ? <Sun className="w-5 h-5 text-cyan-400" /> : <Moon className="w-5 h-5 text-gray-700" />}
            </button>
          </div>
        </div>
      </header>

      
      <div className="relative z-10 pt-40 pb-32 px-8">
        <div className="max-w-[1600px] mx-auto">
        
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-20"
          >
            <h2 className={`text-6xl md:text-7xl font-thin tracking-wider ${isDark ? 'text-white' : 'text-black'} mb-4`}>
              {brandName}
            </h2>
            <div className="h-[1px] w-32 mx-auto" style={{ background: `linear-gradient(90deg, transparent, ${brandAccent}, transparent)` }} />
          </motion.div>

          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-20">
            {models.map((model, index) => {
              const FuelIcon = fuelIcons[model.fuelType];
              const isSelected = selectedModel === model.id;
              
              return (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 60 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ 
                    duration: 0.6, 
                    delay: index * 0.1,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                  whileHover={{ y: -12 }}
                  onClick={() => handleModelSelect(model.id)}
                  className="cursor-pointer group"
                >
                  <Card className={`relative overflow-hidden ${isDark ? 'bg-white/5' : 'bg-gray-50'} backdrop-blur-xl border ${isSelected ? `border-[${brandAccent}]/50` : isDark ? 'border-white/10' : 'border-gray-200'} ${isDark ? 'hover:border-cyan-400/50' : 'hover:border-gray-400'} transition-all duration-500 rounded-2xl ${isSelected ? 'ring-2 ring-offset-2 ring-offset-transparent' : ''}`} style={isSelected ? { ringColor: brandAccent } : {}}>
                    
                    
                    <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" style={{ boxShadow: `inset 0 0 60px ${brandAccent}20` }} />

                    
                    <div className="relative aspect-video overflow-hidden">
                      {model.image ? (
                        
                        <>
                          <img 
                            src={model.image}
                            alt={model.name}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                          />
                          
                          <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
                        </>
                      ) : (
                       
                        <div className={`absolute inset-0 ${isDark ? 'bg-gradient-to-br from-gray-900 via-gray-800 to-black' : 'bg-gradient-to-br from-gray-100 to-gray-200'} group-hover:scale-110 transition-transform duration-700 flex items-center justify-center`}>
                         
                          <div className="relative w-full h-full flex items-center justify-center">
                            
                            <svg className={`w-4/5 h-4/5 ${isDark ? 'text-gray-600' : 'text-gray-400'} group-hover:text-cyan-500/40 transition-colors duration-700`} viewBox="0 0 240 120" fill="none">
                              
                              <ellipse cx="120" cy="105" rx="80" ry="8" fill="currentColor" opacity="0.2" />
                              
                              
                              <path 
                                d="M30 75 L50 50 L90 45 L110 35 L130 45 L170 50 L190 75 L190 85 L30 85 Z" 
                                fill="currentColor" 
                                opacity="0.6"
                                className="group-hover:opacity-80 transition-opacity"
                              />
                              
                             
                              <path 
                                d="M60 50 L70 40 L100 38 L110 35 L120 38 L150 40 L160 50 Z" 
                                fill={isDark ? "#1e293b" : "#cbd5e1"}
                                opacity="0.8"
                              />
                              
                              
                              <circle cx="60" cy="85" r="15" fill="currentColor" opacity="0.8" />
                              <circle cx="60" cy="85" r="10" fill={isDark ? "#334155" : "#94a3b8"} />
                              <circle cx="60" cy="85" r="5" fill="currentColor" opacity="0.4" />
                              
                             
                              <circle cx="160" cy="85" r="15" fill="currentColor" opacity="0.8" />
                              <circle cx="160" cy="85" r="10" fill={isDark ? "#334155" : "#94a3b8"} />
                              <circle cx="160" cy="85" r="5" fill="currentColor" opacity="0.4" />
                              
                              
                              <circle cx="25" cy="70" r="3" fill="#fbbf24" opacity="0.9" className="animate-pulse" />
                              <circle cx="195" cy="70" r="3" fill="#ef4444" opacity="0.7" />
                              
                              
                              <line x1="95" y1="42" x2="125" y2="42" stroke="currentColor" strokeWidth="1" opacity="0.3" />
                              <path d="M30 75 L45 75 L50 70" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
                              <path d="M190 75 L175 75 L170 70" stroke="currentColor" strokeWidth="1.5" opacity="0.4" />
                            </svg>

                            
                            <div 
                              className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700 blur-2xl"
                              style={{ background: `radial-gradient(circle at center, ${brandAccent}40, transparent 70%)` }}
                            />
                          </div>
                        </div>
                      )}


                    </div>

                    
                    <div className="p-6">
                      
                      <h3 className={`text-2xl font-light tracking-wider ${isDark ? 'text-white' : 'text-black'} mb-2 group-hover:text-cyan-400 transition-colors`}>
                        {model.name}
                      </h3>

                     
                      <p className={`text-sm ${isDark ? 'text-white/40' : 'text-gray-500'} mb-4 font-light tracking-wide`}>
                        {model.tagline}
                      </p>

                      
                      <div className="flex items-center gap-2 mb-4">
                        <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs ${isDark ? 'bg-white/5' : 'bg-gray-200'} border ${isDark ? 'border-white/10' : 'border-gray-300'}`}>
                          <FuelIcon className="w-3 h-3" style={{ color: brandAccent }} />
                          <span className={`uppercase tracking-wider ${isDark ? 'text-white/70' : 'text-gray-700'}`}>{model.fuelType}</span>
                        </div>
                      </div>

                      
                      <div className={`grid grid-cols-3 gap-3 p-3 rounded-lg ${isDark ? 'bg-white/5' : 'bg-gray-100'} border ${isDark ? 'border-white/5' : 'border-gray-200'}`}>
                        <div className="text-center">
                          <p className={`text-xs ${isDark ? 'text-white/40' : 'text-gray-500'} mb-1`}>Mileage</p>
                          <p className={`text-sm font-medium ${isDark ? 'text-cyan-400' : 'text-blue-600'}`}>{model.mileage}</p>
                        </div>
                        <div className="text-center border-x ${isDark ? 'border-white/10' : 'border-gray-300'}">
                          <p className={`text-xs ${isDark ? 'text-white/40' : 'text-gray-500'} mb-1`}>Power</p>
                          <p className={`text-sm font-medium ${isDark ? 'text-cyan-400' : 'text-blue-600'}`}>{model.specs.power}</p>
                        </div>
                        <div className="text-center">
                          <p className={`text-xs ${isDark ? 'text-white/40' : 'text-gray-500'} mb-1`}>Price</p>
                          <p className={`text-sm font-medium ${isDark ? 'text-cyan-400' : 'text-blue-600'}`}>{model.priceRange}</p>
                        </div>
                      </div>

                      
                      {isSelected && (
                        <motion.div
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          className="mt-4 py-2 px-4 rounded-full text-center text-sm font-medium"
                          style={{ backgroundColor: `${brandAccent}20`, color: brandAccent }}
                        >
                          ✓ SELECTED
                        </motion.div>
                      )}
                    </div>

                    
                    <div className="absolute top-0 right-0 w-20 h-20 overflow-hidden opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-bl opacity-20" style={{ backgroundImage: `linear-gradient(135deg, ${brandAccent}, transparent)` }} />
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>

          
          <AnimatePresence>
            {selectedModel && (
              <motion.div
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 50 }}
                className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50"
              >
                <button
                  onClick={handleContinue}
                  className={`group flex items-center gap-3 px-8 py-4 rounded-full ${isDark ? 'bg-gradient-to-r from-cyan-500 to-blue-600' : 'bg-gradient-to-r from-blue-500 to-blue-700'} text-white font-medium text-sm tracking-wider shadow-2xl hover:shadow-cyan-500/50 transition-all duration-300 border-2 ${isDark ? 'border-cyan-400/30' : 'border-blue-400/50'}`}
                  style={{ boxShadow: `0 0 40px ${brandAccent}40` }}
                >
                  <span>CONTINUE WITH {selectedModelData?.name.toUpperCase()}</span>
                  <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      
      <style>{`
        @keyframes slide {
          0%, 100% { transform: translateX(-100%); }
          50% { transform: translateX(100vw); }
        }
        @keyframes slideReverse {
          0%, 100% { transform: translateX(100%); }
          50% { transform: translateX(-100vw); }
        }
      `}</style>
    </div>
  );
};

export default ModelSelection;