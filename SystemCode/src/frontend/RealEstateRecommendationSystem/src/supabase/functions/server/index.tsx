import { Hono } from 'npm:hono';
import { cors } from 'npm:hono/cors';
import { logger } from 'npm:hono/logger';
import { createClient } from 'npm:@supabase/supabase-js';
import * as kv from './kv_store.tsx';

const app = new Hono();

// Middleware
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));
app.use('*', logger(console.log));

const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Initialize property data
app.post('/make-server-54aef881/init-properties', async (c) => {
  try {
    const properties = [
      {
        id: "1",
        title: "Modern Family Home",
        price: 650000,
        location: "Oakville, ON",
        bedrooms: 4,
        bathrooms: 3,
        sqft: 2800,
        image: "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBob3VzZSUyMGV4dGVyaW9yfGVufDF8fHx8MTc1Nzg0OTU1Mnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "House",
        features: ["Garage", "Garden", "Updated Kitchen", "Hardwood Floors", "Walk-in Closet"],
        keywords: ["family", "modern", "house", "garage", "garden", "suburban", "spacious", "4 bedroom", "oakville"]
      },
      {
        id: "2",
        title: "Luxury Downtown Condo",
        price: 850000,
        location: "Downtown Toronto, ON",
        bedrooms: 2,
        bathrooms: 2,
        sqft: 1200,
        image: "https://images.unsplash.com/photo-1638454668466-e8dbd5462f20?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxsdXh1cnklMjBhcGFydG1lbnQlMjBpbnRlcmlvcnxlbnwxfHx8fDE3NTc5MDM3NjZ8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "Condo",
        features: ["City View", "Concierge", "Gym", "Pool", "Balcony"],
        keywords: ["luxury", "condo", "downtown", "city", "apartment", "high rise", "amenities", "2 bedroom", "toronto"]
      },
      {
        id: "3",
        title: "Charming Suburban Home",
        price: 425000,
        location: "Mississauga, ON",
        bedrooms: 3,
        bathrooms: 2,
        sqft: 1850,
        image: "https://images.unsplash.com/photo-1570129477492-45c003edd2be?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmYW1pbHklMjBob21lJTIwc3VidXJiYW58ZW58MXx8fHwxNzU3ODUzODc4fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "House",
        features: ["Fireplace", "Deck", "Finished Basement", "Driveway"],
        keywords: ["suburban", "family", "house", "basement", "deck", "affordable", "3 bedroom", "mississauga"]
      },
      {
        id: "4",
        title: "Executive Townhouse",
        price: 780000,
        location: "Markham, ON",
        bedrooms: 3,
        bathrooms: 3,
        sqft: 2200,
        image: "https://images.unsplash.com/photo-1659621222272-f65c27b6f182?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxkb3dudG93biUyMGNvbmRvJTIwY2l0eXxlbnwxfHx8fDE3NTc5MDM3Njd8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "Townhouse",
        features: ["Garage", "Ensuite", "Open Concept", "Granite Counters"],
        keywords: ["townhouse", "executive", "modern", "open concept", "garage", "3 bedroom", "markham"]
      },
      {
        id: "5",
        title: "Starter Condo",
        price: 320000,
        location: "North York, ON",
        bedrooms: 1,
        bathrooms: 1,
        sqft: 650,
        image: "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBob3VzZSUyMGV4dGVyaW9yfGVufDF8fHx8MTc1Nzg0OTU1Mnww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "Condo",
        features: ["Transit Access", "Storage", "Laundry", "Balcony"],
        keywords: ["starter", "condo", "affordable", "transit", "1 bedroom", "first time buyer", "north york"]
      },
      {
        id: "6",
        title: "Luxury Estate",
        price: 1200000,
        location: "Richmond Hill, ON",
        bedrooms: 5,
        bathrooms: 4,
        sqft: 4200,
        image: "https://images.unsplash.com/photo-1570129477492-45c003edd2be?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxmYW1pbHklMjBob21lJTIwc3VidXJiYW58ZW58MXx8fHwxNzU3ODUzODc4fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
        type: "House",
        features: ["Pool", "Wine Cellar", "3-Car Garage", "Home Theatre", "Chef's Kitchen"],
        keywords: ["luxury", "estate", "pool", "large", "expensive", "5 bedroom", "richmond hill"]
      }
    ];

    // Store each property in the KV store
    for (const property of properties) {
      await kv.set(`property:${property.id}`, property);
    }

    console.log('Properties initialized successfully');
    return c.json({ success: true, count: properties.length });
  } catch (error) {
    console.log('Error initializing properties:', error);
    return c.json({ error: 'Failed to initialize properties' }, 500);
  }
});

// Search properties based on user criteria
app.post('/make-server-54aef881/search-properties', async (c) => {
  try {
    const { userMessage } = await c.req.json();
    
    // Get all properties
    const propertyKeys = await kv.getByPrefix('property:');
    const properties = propertyKeys.map(item => item.value);
    
    // Parse user message for criteria
    const message = userMessage.toLowerCase();
    const bedroomMatch = message.match(/(\d+)\s*(?:bed|bedroom)/);
    const budgetMatch = message.match(/(?:under|below|max|maximum|budget)\s*(?:of)?\s*\$?(\d+(?:,\d{3})*(?:k|000)?)/);
    const typeMatch = message.match(/\b(condo|house|townhouse|apartment)\b/);
    
    const preferences = {
      bedrooms: bedroomMatch ? parseInt(bedroomMatch[1]) : null,
      maxBudget: budgetMatch ? parseBudget(budgetMatch[1]) : null,
      type: typeMatch ? typeMatch[1] : null
    };

    // Score and filter properties
    const scoredProperties = properties.map(property => {
      let score = 0;
      
      // Bedroom match
      if (preferences.bedrooms && property.bedrooms === preferences.bedrooms) {
        score += 50;
      } else if (preferences.bedrooms && Math.abs(property.bedrooms - preferences.bedrooms) === 1) {
        score += 25;
      }
      
      // Budget match
      if (preferences.maxBudget && property.price <= preferences.maxBudget) {
        score += 40;
      } else if (preferences.maxBudget && property.price > preferences.maxBudget) {
        score -= 30;
      }
      
      // Type match
      if (preferences.type && property.type.toLowerCase() === preferences.type) {
        score += 30;
      }
      
      // Keyword matching
      property.keywords.forEach(keyword => {
        if (message.includes(keyword)) {
          score += 10;
        }
      });
      
      // Location matching
      if (message.includes(property.location.toLowerCase()) || 
          property.location.toLowerCase().includes(message.split(' ').find(word => word.length > 3) || '')) {
        score += 20;
      }
      
      return { property, score };
    });

    // Return top 3 matches
    const topProperties = scoredProperties
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(item => item.property);

    return c.json({ properties: topProperties, preferences });
  } catch (error) {
    console.log('Error searching properties:', error);
    return c.json({ error: 'Failed to search properties' }, 500);
  }
});

// Save user conversation
app.post('/make-server-54aef881/save-conversation', async (c) => {
  try {
    const { sessionId, message, isUser, properties } = await c.req.json();
    
    const conversationKey = `conversation:${sessionId}`;
    const existingConversation = await kv.get(conversationKey) || [];
    
    const newMessage = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      content: message,
      isUser,
      timestamp: new Date().toISOString(),
      properties: properties || []
    };
    
    existingConversation.push(newMessage);
    await kv.set(conversationKey, existingConversation);
    
    return c.json({ success: true, message: newMessage });
  } catch (error) {
    console.log('Error saving conversation:', error);
    return c.json({ error: 'Failed to save conversation' }, 500);
  }
});

// Get conversation history
app.get('/make-server-54aef881/conversation/:sessionId', async (c) => {
  try {
    const sessionId = c.req.param('sessionId');
    const conversation = await kv.get(`conversation:${sessionId}`) || [];
    
    return c.json({ conversation });
  } catch (error) {
    console.log('Error getting conversation:', error);
    return c.json({ error: 'Failed to get conversation' }, 500);
  }
});

// Generate AI response
app.post('/make-server-54aef881/generate-response', async (c) => {
  try {
    const { userMessage } = await c.req.json();
    const message = userMessage.toLowerCase();
    
    // Get matching properties
    const { properties } = await fetch(`${c.req.url.replace('/generate-response', '/search-properties')}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ userMessage })
    }).then(res => res.json());
    
    // Generate contextual response
    let agentMessage = "";
    let inspirationImages = [];
    
    // Handle comparison requests
    if (message.includes('compare') || message.includes('vs') || message.includes('versus')) {
      agentMessage = `Great question! Let me compare these properties for you:

**Modern Family Home (Oakville) vs Executive Townhouse (Markham):**

🏠 **Space & Layout:**
• Family Home: 4 bed/3 bath, 2,800 sqft - more spacious for larger families
• Townhouse: 3 bed/3 bath, 2,200 sqft - efficient layout with modern design

💰 **Pricing:**
• Family Home: $650,000 - better value per square foot
• Townhouse: $780,000 - premium for executive features

📍 **Location Benefits:**
• Oakville: Established suburban community, great schools
• Markham: Growing tech hub, excellent transit access

✨ **Key Features:**
• Family Home: Garden, garage, traditional charm
• Townhouse: Open concept, granite counters, executive finishes

Both are excellent choices depending on your priorities - space vs. modern amenities!`;
      
      return c.json({ message: agentMessage, properties: properties.slice(0, 2) });
    }
    
    // Handle shopping/amenities requests
    if (message.includes('shopping') || message.includes('mall') || message.includes('food court') || message.includes('restaurant') || message.includes('cafe')) {
      const locationMatch = message.match(/(toronto|oakville|mississauga|markham|north york|richmond hill)/i);
      const location = locationMatch ? locationMatch[1] : 'Toronto';
      
      agentMessage = `Here are the best shopping and dining options near ${location}:

🛍️ **Shopping Centers:**
• Eaton Centre - Premium shopping, 300+ stores
• Yorkdale Shopping Centre - Luxury brands, dining
• Square One (Mississauga) - Largest mall in Ontario
• Markville Shopping Centre - Local favorite

🍽️ **Food Courts & Restaurants:**
• PATH Underground - 50+ restaurants downtown
• Kensington Market - Diverse food scene
• Little Italy - Authentic Italian cuisine
• Chinatown - Amazing Asian food options

🚇 **Transit Access:**
• TTC subway connections to all major malls
• GO Transit for suburban shopping centers
• Walking distance to many local eateries

These locations offer everything from casual dining to fine dining, plus entertainment options!`;
      
      return c.json({ message: agentMessage, properties: [] });
    }
    
    // Handle decoration requests
    if (message.includes('decoration') || message.includes('design') || message.includes('decor')) {
      const roomMatch = message.match(/(living room|bedroom|kitchen|bathroom|dining room)/i);
      const room = roomMatch ? roomMatch[1] : 'living room';
      
      agentMessage = `I'd love to help you design your ${room}! Here are some modern decoration ideas:

🎨 **Color Scheme:**
• Neutral base: Whites, grays, and beiges
• Accent colors: Navy blue, forest green, or warm terracotta
• Metallic touches: Brass or matte black fixtures

🪑 **Furniture & Layout:**
• Multi-functional pieces for smaller spaces
• Mix textures: Velvet, linen, and natural wood
• Create conversation areas with proper spacing

💡 **Lighting:**
• Layer different light sources (ambient, task, accent)
• Add pendant lights or statement chandeliers
• Use warm LED bulbs (2700K-3000K)

🌿 **Finishing Touches:**
• Add plants for natural elements
• Gallery wall with personal artwork
• Cozy textiles: throw pillows, blankets, rugs

Would you like specific product recommendations or help with a particular style theme?`;
      
      // Add some inspiration images
      inspirationImages = [
        {
          id: 'inspiration-1',
          title: 'Modern Living Room Design',
          image: 'https://images.unsplash.com/photo-1720247520862-7e4b14176fa8?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBsaXZpbmclMjByb29tJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzU3ODM4NDk5fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Clean lines, neutral colors, natural lighting'
        },
        {
          id: 'inspiration-2', 
          title: 'Bedroom Interior Design',
          image: 'https://images.unsplash.com/photo-1600210491741-a69593e43133?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWRyb29tJTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzU3ODczNDAwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Cozy textures, warm lighting, minimalist approach'
        },
        {
          id: 'inspiration-3',
          title: 'Modern Kitchen Design', 
          image: 'https://images.unsplash.com/photo-1641823911769-c55f23c25143?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxraXRjaGVuJTIwaW50ZXJpb3IlMjBtb2Rlcm58ZW58MXx8fHwxNzU3OTA1ODE0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Sleek cabinetry, stone countertops, integrated appliances'
        }
      ];
      
      return c.json({ message: agentMessage, properties: [], inspirationImages });
    }
    
    // Handle picture/inspiration requests
    if (message.includes('picture') || message.includes('show me') || message.includes('image') || message.includes('photo')) {
      agentMessage = `Here are some beautiful interior design inspirations for your space:

📸 **Design Inspiration Gallery:**

These images showcase different styles and approaches to modern interior design. Each one demonstrates key principles like:

• **Balance & Proportion** - Furniture scaled appropriately to room size
• **Color Harmony** - Cohesive color palettes that flow naturally  
• **Lighting Strategy** - Multiple light sources creating ambiance
• **Texture Mix** - Combining different materials for visual interest

✨ **Style Tips:**
• Start with a neutral foundation and add personality through accessories
• Invest in quality pieces that will last and grow with your style
• Don't forget about functionality - beautiful spaces should also be livable
• Mix high and low-end pieces for an authentic, curated look

Would you like me to focus on a specific room or style preference for your property?`;

      inspirationImages = [
        {
          id: 'inspiration-1',
          title: 'Modern Living Room Design',
          image: 'https://images.unsplash.com/photo-1720247520862-7e4b14176fa8?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBsaXZpbmclMjByb29tJTIwaW50ZXJpb3J8ZW58MXx8fHwxNzU3ODM4NDk5fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Contemporary living space with clean lines'
        },
        {
          id: 'inspiration-2',
          title: 'Bedroom Interior Design', 
          image: 'https://images.unsplash.com/photo-1600210491741-a69593e43133?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxiZWRyb29tJTIwaW50ZXJpb3IlMjBkZXNpZ258ZW58MXx8fHwxNzU3ODczNDAwfDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Cozy bedroom with natural textures'
        },
        {
          id: 'inspiration-3',
          title: 'Modern Kitchen Design',
          image: 'https://images.unsplash.com/photo-1641823911769-c55f23c25143?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxraXRjaGVuJTIwaW50ZXJpb3IlMjBtb2Rlcm58ZW58MXx8fHwxNzU3OTA1ODE0fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral',
          description: 'Sleek kitchen with premium finishes'
        }
      ];
      
      return c.json({ message: agentMessage, properties: [], inspirationImages });
    }
    
    // Handle difference/comparison questions
    if (message.includes('difference') || message.includes('what\'s the')) {
      agentMessage = `Great question! Here are the key differences between condo and house living:

🏢 **Condo Living:**
✅ **Pros:**
• Lower maintenance - building management handles exterior
• Amenities included (gym, pool, concierge)
• Urban locations with transit access
• Generally more affordable entry point
• Enhanced security features

❌ **Cons:**
• Monthly condo fees
• Less privacy and space
• Parking often costs extra
• Restrictions on renovations
• Shared walls/noise considerations

🏠 **House Living:**
✅ **Pros:**
• Complete privacy and independence
• Your own land and yard space
• No monthly fees beyond utilities
• Freedom to renovate as desired
• Usually more storage and space

❌ **Cons:**
• All maintenance is your responsibility
• Higher upfront costs typically
• May require car for transportation
• Property taxes often higher
• More time-intensive upkeep

The choice depends on your lifestyle, budget, and priorities!`;
      
      return c.json({ message: agentMessage, properties: [] });
    }
    
    // Standard responses for other cases
    if (message.includes('hello') || message.includes('hi') || message.includes('help')) {
      agentMessage = "Hello! I'm your AI real estate agent. I'm here to help you find the perfect property. Please tell me about your preferences - what type of property are you looking for, how many bedrooms, your budget range, and preferred location?";
    } else if (message.includes('budget') || message.includes('price') || message.includes('cost')) {
      const budgetMatch = message.match(/(?:under|below|max|maximum|budget)\s*(?:of)?\s*\$?(\d+(?:,\d{3})*(?:k|000)?)/);
      if (budgetMatch) {
        agentMessage = `I found some great properties within your budget. Here are my top recommendations that match your criteria:`;
      } else {
        agentMessage = "What's your budget range? This will help me find properties that are right for you. For example, you could say 'I'm looking for something under $500,000' or 'My budget is around $800k'.";
      }
    } else if (message.includes('bedroom') || message.includes('bed')) {
      const bedroomMatch = message.match(/(\d+)\s*(?:bed|bedroom)/);
      if (bedroomMatch) {
        agentMessage = `Perfect! I found some excellent ${bedroomMatch[1]}-bedroom properties for you. Here are my top recommendations:`;
      }
    } else if (message.includes('condo') || message.includes('apartment')) {
      agentMessage = "Great choice! Condos offer excellent amenities and urban convenience. Here are some condos I think you'll love:";
    } else if (message.includes('house') || message.includes('home')) {
      agentMessage = "Houses are perfect for families and those who want more space. Here are some wonderful houses that match your criteria:";
    } else if (properties.length > 0) {
      agentMessage = "Based on your requirements, I've found some properties that might interest you. Here are my top recommendations:";
    } else {
      agentMessage = "I'd love to help you find the perfect property! Could you tell me more about what you're looking for? For example:\n\n• How many bedrooms do you need?\n• What's your budget range?\n• Do you prefer a house, condo, or townhouse?\n• Any specific neighborhood or area?\n• Any special features you're looking for?";
    }
    
    return c.json({ message: agentMessage, properties, inspirationImages });
  } catch (error) {
    console.log('Error generating response:', error);
    return c.json({ error: 'Failed to generate response' }, 500);
  }
});

function parseBudget(budgetStr: string): number {
  const cleanStr = budgetStr.replace(/,/g, '');
  if (cleanStr.endsWith('k')) {
    return parseInt(cleanStr.slice(0, -1)) * 1000;
  } else if (cleanStr.endsWith('000')) {
    return parseInt(cleanStr);
  } else {
    return parseInt(cleanStr);
  }
}

// Health check
app.get('/make-server-54aef881/health', (c) => {
  return c.json({ status: 'ok', timestamp: new Date().toISOString() });
});

Deno.serve(app.fetch);