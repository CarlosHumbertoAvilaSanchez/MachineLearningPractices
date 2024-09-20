import { defineCollection, z } from "astro:content";


const activities = defineCollection({
    schema: z.object({
        fileName: z.string(),
        title: z.string(),
        student: z.string(),
        school: z.string(),
        subject: z.string(),
        teacher: z.string(),
        classSection: z.string(),
    }),
})

export const collections = { activities };